"""
ml/dataset_carimbo.py
======================
Preparação do dataset para treinar um detector de carimbo baseado em
YOLO (Ultralytics).

Formato de entrada esperado (o mesmo produzido por ferramentas de
anotação comuns como LabelImg, CVAT ou Label Studio ao exportar em
"YOLO format"):

    pasta_anotacoes/
        prancha_0001.tif   (ou .jpg/.png — a imagem original)
        prancha_0001.txt   (uma linha por carimbo: "0 xc yc w h", normalizado 0-1)
        prancha_0002.tif
        prancha_0002.txt
        ...

Só existe uma classe ("carimbo", índice 0). Cada imagem pode ter mais
de um carimbo anotado (raro, mas suportado).

Este módulo organiza os pares imagem/anotação em uma estrutura de
diretórios compatível com o treinamento do Ultralytics YOLO:

    dataset_saida/
        images/train/...
        images/val/...
        labels/train/...
        labels/val/...
        data.yaml
"""

from __future__ import annotations

import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("campvision.ml.dataset_carimbo")

EXTENSOES_IMAGEM = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
NOME_CLASSE = "carimbo"


@dataclass
class ResumoDataset:
    total_pares: int
    total_treino: int
    total_validacao: int
    caminho_yaml: Path


def _encontrar_pares(pasta_anotacoes: Path) -> list[tuple[Path, Path]]:
    """Localiza pares (imagem, anotação .txt) na pasta de anotações."""
    pares = []
    for anotacao in sorted(pasta_anotacoes.glob("*.txt")):
        imagem = None
        for ext in EXTENSOES_IMAGEM:
            candidata = anotacao.with_suffix(ext)
            if candidata.exists():
                imagem = candidata
                break
        if imagem is None:
            logger.warning("Anotação sem imagem correspondente: %s", anotacao.name)
            continue
        pares.append((imagem, anotacao))
    return pares


def preparar_dataset(
    pasta_anotacoes: Path,
    pasta_saida: Path,
    proporcao_validacao: float = 0.15,
    semente_aleatoria: int = 42,
) -> ResumoDataset:
    """Organiza imagens e anotações anotadas manualmente em uma
    estrutura pronta para o treinamento YOLO, com split treino/val."""
    pares = _encontrar_pares(pasta_anotacoes)
    if not pares:
        raise ValueError(f"Nenhum par imagem/anotação encontrado em {pasta_anotacoes}")

    random.Random(semente_aleatoria).shuffle(pares)
    ponto_corte = max(1, int(len(pares) * (1 - proporcao_validacao)))
    pares_treino = pares[:ponto_corte]
    pares_validacao = pares[ponto_corte:] or pares[-1:]  # garante ao menos 1 exemplo de val

    for subconjunto, itens in (("train", pares_treino), ("val", pares_validacao)):
        (pasta_saida / "images" / subconjunto).mkdir(parents=True, exist_ok=True)
        (pasta_saida / "labels" / subconjunto).mkdir(parents=True, exist_ok=True)
        for imagem, anotacao in itens:
            shutil.copy2(imagem, pasta_saida / "images" / subconjunto / imagem.name)
            shutil.copy2(anotacao, pasta_saida / "labels" / subconjunto / anotacao.name)

    caminho_yaml = pasta_saida / "data.yaml"
    caminho_yaml.write_text(
        f"path: {pasta_saida.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: {NOME_CLASSE}\n",
        encoding="utf-8",
    )

    logger.info(
        "Dataset preparado: %d treino, %d validação, yaml em %s",
        len(pares_treino), len(pares_validacao), caminho_yaml,
    )
    return ResumoDataset(
        total_pares=len(pares),
        total_treino=len(pares_treino),
        total_validacao=len(pares_validacao),
        caminho_yaml=caminho_yaml,
    )
