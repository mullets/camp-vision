"""
ml/detector_carimbo_ml.py
===========================
Inferência do detector de carimbo treinado (YOLO/Ultralytics),
expondo a mesma interface de resultado usada pela heurística
(`scanner.detector_carimbo.CarimboDetectado`), para que o restante do
pipeline não precise saber qual estratégia está em uso.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from scanner.detector_carimbo import CarimboDetectado

logger = logging.getLogger("campvision.ml.detector_carimbo")


def _canto_mais_proximo(x1: float, y1: float, x2: float, y2: float, largura_img: int, altura_img: int) -> str:
    """Rotula a caixa detectada com o canto mais próximo, apenas para
    fins de log/depuração — mantém compatibilidade com o campo
    `canto` do resultado heurístico."""
    centro_x, centro_y = (x1 + x2) / 2, (y1 + y2) / 2
    metade_x, metade_y = largura_img / 2, altura_img / 2
    vertical = "superior" if centro_y < metade_y else "inferior"
    horizontal = "esquerdo" if centro_x < metade_x else "direito"
    return f"{vertical}_{horizontal}"


def _detectar_tamanho_imagem_treino(modelo, padrao: int = 960) -> int:
    """Lê o tamanho de imagem usado no treino direto do checkpoint do
    modelo — o Ultralytics guarda os argumentos de treino (incluindo
    `imgsz`) dentro do próprio arquivo `.pt`. Assim o usuário não
    precisa saber nem informar esse número manualmente: cada modelo
    treinado já carrega essa informação consigo.

    Tenta alguns lugares onde essa informação costuma aparecer,
    dependendo da versão do Ultralytics; se não achar em nenhum,
    registra um aviso e usa `padrao`."""
    candidatos = []

    try:
        candidatos.append(modelo.overrides.get("imgsz"))
    except Exception:  # noqa: BLE001
        pass
    try:
        candidatos.append(modelo.ckpt.get("train_args", {}).get("imgsz"))
    except Exception:  # noqa: BLE001
        pass
    try:
        candidatos.append(modelo.ckpt["train_args"]["imgsz"])
    except Exception:  # noqa: BLE001
        pass
    try:
        candidatos.append(getattr(modelo.model, "args", {}).get("imgsz"))
    except Exception:  # noqa: BLE001
        pass

    for valor in candidatos:
        if isinstance(valor, (int, float)) and valor > 0:
            return int(valor)
        if isinstance(valor, (list, tuple)) and valor and isinstance(valor[0], (int, float)) and valor[0] > 0:
            return int(valor[0])

    logger.warning(
        "Não consegui ler o tamanho de imagem do treino direto do arquivo do modelo — "
        "usando %d como padrão. Se a detecção estiver ruim, é possível informar o valor "
        "manualmente em Configurações (\"Tamanho de imagem\").",
        padrao,
    )
    return padrao


class DetectorCarimboML:
    """Wrapper de inferência de um modelo YOLO treinado para localizar
    o carimbo. Carrega os pesos uma única vez por instância."""

    def __init__(self, caminho_modelo: str, confianca_minima: float = 0.5, tamanho_imagem: Optional[int] = None):
        from ultralytics import YOLO  # import tardio: dependência pesada e opcional

        logger.info("Carregando modelo de detecção de carimbo: %s", caminho_modelo)
        self._modelo = YOLO(caminho_modelo)
        self.confianca_minima = confianca_minima

        # IMPORTANTE: o tamanho de imagem da inferência precisa bater
        # com o usado no treino — um valor diferente (ex.: o padrão
        # 640 do Ultralytics para inferência, quando o treino usou
        # 960) reduz a nitidez efetiva do carimbo dentro da prancha
        # gigante, prejudicando a detecção. Em vez de exigir que o
        # usuário informe esse número manualmente, lemos direto do
        # checkpoint do modelo (`tamanho_imagem=None`, o padrão); só
        # usamos um valor explícito se ele for passado por fora (ex.:
        # ajuste manual nas Configurações).
        if tamanho_imagem is None:
            self.tamanho_imagem = _detectar_tamanho_imagem_treino(self._modelo)
            logger.info("Tamanho de imagem detectado automaticamente do modelo: %d", self.tamanho_imagem)
        else:
            self.tamanho_imagem = tamanho_imagem

        # "Aquece" o modelo com uma inferência dummy, de forma síncrona,
        # ANTES de ser compartilhado entre as threads do lote. O
        # Ultralytics faz uma otimização de camadas (fusão) de forma
        # preguiçosa na primeira chamada de predict() — se várias
        # threads chamarem predict() ao mesmo tempo nesse modelo
        # recém-carregado, cada uma pode disparar essa fusão de forma
        # concorrente e não sincronizada, o que é uma condição de
        # corrida real (não só um print repetido no log: risco de
        # corromper o modelo). Fazendo essa primeira chamada aqui, de
        # forma síncrona, garantimos que a fusão já aconteceu antes de
        # qualquer thread ter acesso ao modelo.
        imagem_aquecimento = np.zeros((self.tamanho_imagem, self.tamanho_imagem, 3), dtype=np.uint8)
        self._modelo.predict(imagem_aquecimento, verbose=False, imgsz=self.tamanho_imagem, conf=0.05)
        logger.debug("Modelo de carimbo aquecido (fusão de camadas concluída).")

    def detectar(self, imagem: np.ndarray) -> Optional[CarimboDetectado]:
        """Executa a inferência sobre a imagem (BGR) e retorna a
        detecção de maior confiança, ou None se nenhuma ultrapassar o
        limiar mínimo configurado."""
        altura_img, largura_img = imagem.shape[:2]
        # IMPORTANTE: passamos conf=0.05 aqui (bem permissivo) em vez do
        # padrão do Ultralytics (0.25) — assim TODAS as detecções
        # relevantes chegam até aqui, e a única decisão real de "aceitar
        # ou não" é o `self.confianca_minima` configurado pelo usuário.
        # Sem isso, havia dois filtros empilhados (o interno do
        # Ultralytics E o nosso), o que confundia o diagnóstico: uma
        # detecção podia estar sendo descartada silenciosamente pelo
        # padrão de 0.25 antes mesmo de chegarmos a comparar com o
        # limiar configurado (ex.: 0.80).
        resultados = self._modelo.predict(imagem, verbose=False, imgsz=self.tamanho_imagem, conf=0.05)

        if not resultados or resultados[0].boxes is None or len(resultados[0].boxes) == 0:
            logger.info("Modelo treinado não encontrou nenhum carimbo, nem em confiança baixa (0.05).")
            return None

        caixas = resultados[0].boxes
        melhor_indice = int(caixas.conf.argmax())
        confianca = float(caixas.conf[melhor_indice])

        if confianca < self.confianca_minima:
            # Nível INFO de propósito (não DEBUG): saber que o modelo
            # ACHOU algo mas foi rejeitado só pelo limiar de confiança
            # é um diagnóstico importante — indica que baixar
            # "Confiança mínima" nas Configurações pode resolver, sem
            # precisar de mais treino.
            logger.info("Carimbo candidato encontrado (confiança %.2f) mas abaixo do limiar mínimo (%.2f) — ignorado.",
                        confianca, self.confianca_minima)
            return None

        x1, y1, x2, y2 = [float(v) for v in caixas.xyxy[melhor_indice].tolist()]
        canto = _canto_mais_proximo(x1, y1, x2, y2, largura_img, altura_img)

        return CarimboDetectado(
            x=max(0, int(x1)),
            y=max(0, int(y1)),
            largura=max(1, int(x2 - x1)),
            altura=max(1, int(y2 - y1)),
            confianca=confianca,
            canto=canto,
        )
