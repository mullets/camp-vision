"""
ocr/paddle_ocr.py
==================
Módulo independente de OCR, baseado no PaddleOCR.

Isolado do restante do sistema por trás de uma interface simples
(`extrair_texto`) para que o motor de OCR possa ser trocado no futuro
(ex.: Tesseract, cloud OCR) apenas implementando a mesma interface,
sem alterar o restante do pipeline.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache

import numpy as np

from ocr.base import ResultadoOCR

logger = logging.getLogger("campvision.ocr")

_trava_inicializacao = threading.Lock()


@lru_cache(maxsize=1)
def _obter_engine(idiomas: tuple[str, ...] = ("pt", "en")):
    """Instancia o motor PaddleOCR uma única vez (cache), pois o
    carregamento do modelo é custoso. O idioma principal é 'pt' já
    que as pranchas são majoritariamente em português.

    A inicialização é protegida por um lock (`_trava_inicializacao`):
    se várias threads chamarem esta função ao mesmo tempo antes do
    cache ser preenchido (`lru_cache` só protege o dicionário de
    cache em si, não o corpo da função), cada uma disparava sua
    própria inicialização pesada e redundante — o mesmo tipo de
    problema já visto com o modelo treinado de carimbo.

    Também tenta a inicialização com um conjunto reduzido de
    argumentos se a chamada "completa" falhar — versões diferentes do
    PaddleOCR mudam a API (ex.: `show_log` foi removido numa versão
    mais nova), então preferimos degradar graciosamente a quebrar o
    OCR inteiro por causa de um argumento que uma versão específica
    não reconhece mais."""
    from paddleocr import PaddleOCR  # import tardio: dependência pesada

    idioma_principal = idiomas[0] if idiomas else "pt"

    with _trava_inicializacao:
        logger.info("Inicializando PaddleOCR (idioma=%s)...", idioma_principal)
        try:
            return PaddleOCR(use_angle_cls=True, lang=idioma_principal, show_log=False)
        except (ValueError, TypeError) as exc:
            logger.warning(
                "PaddleOCR não aceitou os argumentos completos (%s) — tentando um conjunto "
                "reduzido, compatível com versões mais novas da biblioteca.", exc,
            )
        try:
            return PaddleOCR(use_angle_cls=True, lang=idioma_principal)
        except (ValueError, TypeError) as exc:
            logger.warning(
                "PaddleOCR ainda não aceitou (%s) — tentando só com o idioma.", exc,
            )
        return PaddleOCR(lang=idioma_principal)


_paddle_indisponivel = False  # vira True se a instalação estiver quebrada nesta máquina


def extrair_texto(imagem: np.ndarray, idiomas: list[str] | None = None) -> ResultadoOCR:
    """Executa OCR sobre a imagem (array BGR) e retorna texto, bounding
    boxes e confiança por linha, além da confiança média geral.

    Se o PaddleOCR estiver quebrado nesta instalação (incompatibilidade
    interna da biblioteca, não algo que possamos corrigir aqui), cai
    automaticamente para o Tesseract pelo resto da execução — em vez de
    repetir o mesmo erro pesado a cada prancha do lote."""
    global _paddle_indisponivel

    idiomas_tupla = tuple(idiomas or ["pt", "en"])

    if _paddle_indisponivel:
        from ocr.tesseract_ocr import extrair_texto as extrair_texto_tesseract
        return extrair_texto_tesseract(imagem, list(idiomas_tupla))

    try:
        engine = _obter_engine(idiomas_tupla)
    except Exception as exc:  # noqa: BLE001
        _paddle_indisponivel = True
        logger.error(
            "PaddleOCR não pôde ser inicializado nesta instalação (%s). "
            "Usando Tesseract pelo resto desta execução — para não ver este aviso, "
            "troque o motor de OCR para 'tesseract' nas Configurações.", exc,
        )
        from ocr.tesseract_ocr import extrair_texto as extrair_texto_tesseract
        return extrair_texto_tesseract(imagem, list(idiomas_tupla))

    resultado_bruto = engine.ocr(imagem, cls=True)

    linhas: list[str] = []
    boxes: list[list[list[float]]] = []
    confidences: list[float] = []

    # A saída do PaddleOCR é uma lista (por imagem) de listas de
    # [ [ [x,y]*4 ], (texto, confianca) ]
    paginas = resultado_bruto or []
    for pagina in paginas:
        if not pagina:
            continue
        for deteccao in pagina:
            poligono, (texto, confianca) = deteccao
            linhas.append(texto)
            boxes.append(poligono)
            confidences.append(float(confianca))

    confianca_media = float(np.mean(confidences)) if confidences else 0.0
    texto_completo = "\n".join(linhas)

    logger.debug("OCR extraiu %d linhas, confiança média=%.2f", len(linhas), confianca_media)

    return ResultadoOCR(
        texto=texto_completo,
        linhas=linhas,
        bounding_boxes=boxes,
        confidences=confidences,
        confianca_media=confianca_media,
    )
