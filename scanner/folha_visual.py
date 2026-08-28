"""
scanner/folha_visual.py
=========================
Leitura do número da FOLHA quando ele não aparece no texto do OCR.

Em muitos carimbos de arquitetura o número da folha não é escrito
como texto comum ao lado do rótulo "FOLHA": é um algarismo GRANDE
desenhado à mão em contorno vazado (oco), ocupando uma faixa própria
à direita do carimbo. O Tesseract lê muito mal esse tipo de forma —
ele espera letras sólidas, não um contorno.

A estratégia aqui é tratar o algarismo como DESENHO antes de tratá-lo
como texto: isolar a faixa à direita, achar o maior contorno isolado
(que é o algarismo), PREENCHÊ-LO para virar uma forma sólida, e só
então passar ao OCR em modo "caractere único". Testado com carimbos
reais do acervo (praças SABESP-COGEP): 5 de 5 folhas lidas
corretamente, contra 0 pela leitura de texto comum.

O número da folha importa porque é o que distingue uma prancha da
outra dentro do mesmo projeto — todo o resto do carimbo é idêntico
entre as folhas.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("campvision.folha")

# Fração à direita do carimbo onde o algarismo da folha costuma ficar.
PROPORCAO_FAIXA_DIREITA = 0.75

# Altura mínima do algarismo, como fração da altura do carimbo — evita
# confundir com sujeira, pontuação ou um pedaço de linha da moldura.
ALTURA_MINIMA_RELATIVA = 0.08

# Acima disso o "algarismo" provavelmente é a própria moldura do
# carimbo ou uma linha longa, não um número.
ALTURA_MAXIMA_RELATIVA = 0.95

MARGEM_RECORTE = 8
MARGEM_BRANCA = 25


def ler_numero_folha(recorte_carimbo: np.ndarray) -> Optional[str]:
    """Tenta ler o número da folha desenhado em contorno na faixa
    direita do carimbo. Retorna o número como string, ou None se não
    for possível identificá-lo com confiança."""
    try:
        return _ler(recorte_carimbo)
    except Exception as exc:  # noqa: BLE001
        # Esta é uma tentativa AUXILIAR: se falhar, o pipeline segue
        # normalmente com a folha vazia. Nunca deve derrubar nada.
        logger.debug("Falha ao ler número da folha visualmente: %s", exc)
        return None


def _ler(recorte_carimbo: np.ndarray) -> Optional[str]:
    import pytesseract

    if recorte_carimbo is None or recorte_carimbo.size == 0:
        return None

    cinza = (
        cv2.cvtColor(recorte_carimbo, cv2.COLOR_BGR2GRAY)
        if recorte_carimbo.ndim == 3 else recorte_carimbo
    )
    altura, largura = cinza.shape
    faixa = cinza[:, int(largura * PROPORCAO_FAIXA_DIREITA):]
    if faixa.size == 0:
        return None

    _, binaria = cv2.threshold(faixa, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None

    maior = max(contornos, key=cv2.contourArea)
    x, y, largura_num, altura_num = cv2.boundingRect(maior)

    proporcao = altura_num / faixa.shape[0]
    if not (ALTURA_MINIMA_RELATIVA <= proporcao <= ALTURA_MAXIMA_RELATIVA):
        return None

    # Preenche o contorno: o algarismo é desenhado oco, e o OCR só
    # reconhece a forma depois de sólida.
    preenchido = np.zeros(binaria.shape, np.uint8)
    cv2.drawContours(preenchido, [maior], -1, 255, thickness=cv2.FILLED)

    recorte = preenchido[
        max(0, y - MARGEM_RECORTE): y + altura_num + MARGEM_RECORTE,
        max(0, x - MARGEM_RECORTE): x + largura_num + MARGEM_RECORTE,
    ]
    if recorte.size == 0:
        return None

    # OCR espera texto escuro sobre fundo claro, com folga em volta.
    recorte = cv2.bitwise_not(recorte)
    recorte = cv2.copyMakeBorder(
        recorte, MARGEM_BRANCA, MARGEM_BRANCA, MARGEM_BRANCA, MARGEM_BRANCA,
        cv2.BORDER_CONSTANT, value=255,
    )

    for psm in (10, 8, 7):  # 10 = caractere único, o caso mais comum aqui
        texto = pytesseract.image_to_string(
            recorte, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789",
        ).strip()
        if texto.isdigit():
            numero = texto.lstrip("0") or "0"
            logger.info("Número da folha lido do desenho do carimbo: %s", numero)
            return numero

    return None
