"""Testes do pré-processamento de imagem antes do OCR
(ocr/tesseract_ocr._preparar_para_ocr) — reduzir granulação/ruído,
mais comum em pranchas UNIDAS (duas metades escaneadas separadamente
e juntadas numa imagem só)."""

import numpy as np
import cv2
import pytest

from ocr.tesseract_ocr import tesseract_disponivel, extrair_texto, _preparar_para_ocr

pytestmark = pytest.mark.skipif(not tesseract_disponivel(), reason="tesseract não instalado neste ambiente")


def _imagem_com_texto_e_ruido_forte(seed: int = 1) -> np.ndarray:
    """Reproduz o cenário reportado: texto legível na origem, mas
    degradado por ruído gaussiano forte + sal-e-pimenta + gradiente de
    iluminação (como aconteceria juntando duas metades escaneadas em
    condições diferentes)."""
    limpa = np.full((300, 900, 3), 255, dtype=np.uint8)
    cv2.putText(limpa, "ARQUITETO CARMONA", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    cv2.putText(limpa, "ESCALA 1 100 PROJETO 4521", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(limpa, "ELETROPAULO SAO PAULO", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    rng = np.random.RandomState(seed)
    gradiente = np.tile(np.linspace(-25, 25, limpa.shape[1]), (limpa.shape[0], 1)).astype(np.int16)
    gradiente = np.stack([gradiente] * 3, axis=-1)
    base = limpa.astype(np.int16) + gradiente
    ruido = rng.normal(0, 65, limpa.shape).astype(np.int16)
    granulada = np.clip(base + ruido, 0, 255).astype(np.uint8)
    mascara = rng.random_sample(limpa.shape[:2])
    granulada[mascara < 0.035] = 0
    granulada[mascara > 0.965] = 255
    return cv2.GaussianBlur(granulada, (3, 3), 1.0)


def test_preparar_para_ocr_devolve_imagem_em_escala_de_cinza():
    granulada = _imagem_com_texto_e_ruido_forte()
    tratada = _preparar_para_ocr(granulada)
    assert tratada.ndim == 2
    assert tratada.shape[:2] == granulada.shape[:2]


def test_pre_processamento_recupera_texto_perdido_por_ruido_forte():
    """Reproduz o problema relatado: sem tratamento, o Tesseract perde
    uma linha inteira de texto numa imagem bem degradada; com o
    pré-processamento (mediana + denoise + Otsu), as três linhas saem
    corretas."""
    granulada = _imagem_com_texto_e_ruido_forte()

    resultado = extrair_texto(granulada, ["en"])

    texto = resultado.texto.upper()
    assert "ARQUITETO" in texto
    assert "CARMONA" in texto
    assert "ELETROPAULO" in texto
    assert "SAO PAULO" in texto
