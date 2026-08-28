"""
ml/dataset_classificacao.py
=============================
Validação do dataset usado para treinar o classificador de tipo de
prancha por imagem (miniatura completa da prancha, não o carimbo).

Formato esperado (compatível com `torchvision.datasets.ImageFolder`):

    pasta_dataset/
        Planta/
            prancha_0001.jpg
            prancha_0002.jpg
            ...
        Corte/
            prancha_0010.jpg
            ...
        Fachada/
            ...
        ...

O nome de cada subpasta deve corresponder a uma das categorias em
`classificacao.classificador.CATEGORIAS`. Este módulo apenas valida a
estrutura e relata contagens por classe (útil para identificar
classes com poucos exemplos antes de treinar); o carregamento real
do dataset para treino é feito por `torchvision.datasets.ImageFolder`
dentro de `ml/treinar_classificador.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from classificacao.classificador import CATEGORIAS

logger = logging.getLogger("campvision.ml.dataset_classificacao")

EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png"}
MINIMO_EXEMPLOS_RECOMENDADO = 30


@dataclass
class ResumoValidacao:
    contagem_por_classe: dict[str, int]
    classes_ausentes: list[str]
    classes_com_poucos_exemplos: list[str]
    total_imagens: int

    def valido(self) -> bool:
        return self.total_imagens > 0 and not self.classes_ausentes


def validar_dataset(pasta_dataset: Path) -> ResumoValidacao:
    """Verifica a estrutura de pastas por classe e contabiliza quantas
    imagens existem em cada uma, sinalizando classes ausentes ou com
    poucos exemplos (o que tende a prejudicar o treinamento)."""
    contagem: dict[str, int] = {}

    for categoria in CATEGORIAS:
        pasta_categoria = pasta_dataset / categoria
        if not pasta_categoria.is_dir():
            contagem[categoria] = 0
            continue
        imagens = [p for p in pasta_categoria.iterdir() if p.suffix.lower() in EXTENSOES_IMAGEM]
        contagem[categoria] = len(imagens)

    classes_ausentes = [c for c, n in contagem.items() if n == 0]
    classes_com_poucos_exemplos = [
        c for c, n in contagem.items() if 0 < n < MINIMO_EXEMPLOS_RECOMENDADO
    ]
    total_imagens = sum(contagem.values())

    if classes_ausentes:
        logger.warning("Classes sem nenhum exemplo no dataset: %s", ", ".join(classes_ausentes))
    if classes_com_poucos_exemplos:
        logger.warning(
            "Classes com poucos exemplos (< %d): %s",
            MINIMO_EXEMPLOS_RECOMENDADO, ", ".join(classes_com_poucos_exemplos),
        )
    logger.info("Total de imagens no dataset: %d", total_imagens)

    return ResumoValidacao(
        contagem_por_classe=contagem,
        classes_ausentes=classes_ausentes,
        classes_com_poucos_exemplos=classes_com_poucos_exemplos,
        total_imagens=total_imagens,
    )
