"""
ocr/motor.py
=============
Fábrica de estratégia para o motor de OCR, seguindo o mesmo padrão
usado para o detector de carimbo e o classificador de tipo (ver
`scanner/detector_carimbo.criar_detector` e
`classificacao/classificador.criar_classificador`).

- "tesseract" (padrão): leve, sem exigências especiais de CPU — ver
  `ocr/tesseract_ocr.py`.
- "paddleocr": maior precisão em geral, mas requer PaddlePaddle
  instalado e, tipicamente, CPU com suporte a AVX/AVX2 — recomendado
  apenas em hardware mais novo.

Se o motor escolhido não estiver disponível por qualquer motivo
(binário/pacote ausente, falha ao carregar), o sistema cai
automaticamente para o Tesseract, para nunca interromper o lote.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Callable

import numpy as np

from ocr.base import ResultadoOCR

logger = logging.getLogger("campvision.ocr.motor")


def criar_motor_ocr(nome: str = "tesseract") -> Callable[[np.ndarray, list], ResultadoOCR]:
    """Retorna a função `extrair_texto(imagem, idiomas)` do motor de
    OCR configurado, com fallback automático para o Tesseract."""
    if nome == "paddleocr":
        # O wrapper (ocr/paddle_ocr.py) importa o PaddleOCR de forma
        # tardia (só na primeira chamada), então checamos aqui se o
        # pacote está de fato instalado — senão o fallback só
        # aconteceria no meio do processamento do primeiro arquivo.
        if importlib.util.find_spec("paddleocr") is None:
            logger.error("Pacote 'paddleocr' não instalado (ver requirements-ml.txt). Usando Tesseract.")
        else:
            try:
                from ocr.paddle_ocr import extrair_texto as extrair_texto_paddle
                logger.info("Motor de OCR: PaddleOCR")
                return extrair_texto_paddle
            except Exception as exc:  # noqa: BLE001
                logger.error("Falha ao preparar PaddleOCR (%s). Usando Tesseract.", exc)

    from ocr.tesseract_ocr import extrair_texto as extrair_texto_tesseract
    logger.info("Motor de OCR: Tesseract")
    return extrair_texto_tesseract
