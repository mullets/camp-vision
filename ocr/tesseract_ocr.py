"""
ocr/tesseract_ocr.py
======================
Módulo de OCR baseado no Tesseract (via `pytesseract`), motor padrão
do CAMP Vision.

Por que Tesseract em vez de PaddleOCR como padrão: o PaddleOCR roda
sobre o PaddlePaddle, cujos pacotes pré-compilados geralmente exigem
instruções de CPU AVX/AVX2 (e frequentemente aceleração por GPU) —
Macs mais antigos (ex.: Mac Pro 5,1 com Xeon Westmere, ou 6,1 com
Xeon Ivy Bridge-EP) não têm AVX2 e podem nem ter AVX, o que faz o
PaddlePaddle falhar silenciosamente ou travar com "illegal
instruction". O Tesseract não tem essa exigência e roda bem em
hardware antigo, sendo por isso o motor recomendado por padrão.

Requer o binário `tesseract` instalado no sistema (não é um pacote
Python):

    brew install tesseract tesseract-lang   # tesseract-lang traz o pt

O PaddleOCR continua disponível como alternativa opcional para quem
tem hardware mais novo e quer maior precisão — ver `ocr/motor.py` e
`ocr/paddle_ocr.py`.
"""

from __future__ import annotations

import logging
import shutil
import numpy as np

from ocr.base import ResultadoOCR

logger = logging.getLogger("campvision.ocr.tesseract")

_MAPA_IDIOMAS = {"pt": "por", "en": "eng", "es": "spa"}


_tesseract_confirmado = False  # só vira True permanentemente; um "não encontrado" nunca é definitivo


def tesseract_disponivel() -> bool:
    """Verifica se o binário `tesseract` está instalado.

    IMPORTANTE: uma vez confirmado disponível, o resultado positivo é
    guardado em cache pro resto da execução (não muda de ideia sem
    motivo). Mas um resultado NEGATIVO nunca é guardado — se checarmos
    e não achar, tentamos de novo na próxima chamada. Isso evita que
    uma falha passageira e pontual (ex.: uma race condition ao checar
    o PATH logo no início do lote, com várias threads arrancando ao
    mesmo tempo) desative o OCR silenciosamente para o resto de um
    lote inteiro — foi exatamente isso que aconteceu num teste real:
    o tesseract falhou em checar disponibilidade uma única vez, e o
    cache antigo (`lru_cache`) guardava esse "não" pra sempre, mesmo
    com o binário disponível e funcionando normalmente antes e
    depois."""
    global _tesseract_confirmado
    if _tesseract_confirmado:
        return True

    disponivel = shutil.which("tesseract") is not None
    if disponivel:
        _tesseract_confirmado = True
        return True

    logger.error(
        "Binário 'tesseract' não encontrado no PATH. Instale com "
        "'brew install tesseract tesseract-lang' para habilitar o OCR."
    )
    return False


def _converter_idiomas(idiomas: list[str] | None) -> str:
    """Converte códigos curtos (ex. 'pt') para os códigos de idioma do
    Tesseract (ex. 'por'), combinando múltiplos idiomas com '+'."""
    idiomas = idiomas or ["pt"]
    codigos = [_MAPA_IDIOMAS.get(i, i) for i in idiomas]
    return "+".join(codigos)


def _preparar_para_ocr(imagem: np.ndarray) -> np.ndarray:
    """Reduz ruído de granulação antes do OCR — mais comum em TIFFs
    digitalizados de acervo, e ainda mais acentuado em pranchas UNIDAS
    (duas metades escaneadas separadamente e juntadas em uma imagem
    só, às vezes com exposição/contraste levemente diferentes entre
    as metades).

    Ordem importa, testada empiricamente com ruído sintético forte
    (gaussiano + sal-e-pimenta + gradiente de iluminação):
      1. Mediana (3x3): sozinha já resolve a maior parte do
         sal-e-pimenta — é o passo de maior impacto. Um blur genérico
         (gaussiano) NÃO faz o mesmo papel aqui: borra o ruído junto
         com o texto em vez de remover os pontos isolados.
      2. fastNlMeansDenoising: limpa o ruído gaussiano residual sem
         borrar as bordas das letras (diferente de um blur comum).
      3. Otsu: binariza com um limiar calculado automaticamente pela
         própria imagem — o Tesseract lê texto preto/branco bem
         definido de forma mais consistente do que cinza ruidoso ou
         com iluminação desigual entre as metades da prancha.

    IMPORTANTE (achado nos testes): NÃO usar limiar adaptativo
    (`adaptiveThreshold`) aqui — ele amplifica ruído sal-e-pimenta
    residual em manchas com cara de texto, piorando o resultado em
    vez de melhorar."""
    import cv2

    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) if imagem.ndim == 3 else imagem
    sem_sal_e_pimenta = cv2.medianBlur(cinza, 3)
    suavizada = cv2.fastNlMeansDenoising(sem_sal_e_pimenta, h=10, templateWindowSize=7, searchWindowSize=21)
    _, binarizada = cv2.threshold(suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarizada


def extrair_texto(imagem: np.ndarray, idiomas: list[str] | None = None) -> ResultadoOCR:
    """Executa OCR sobre a imagem (array BGR) usando Tesseract e
    retorna texto, bounding boxes e confiança por linha, no mesmo
    formato usado pelo restante do pipeline (`ResultadoOCR`)."""
    if not tesseract_disponivel():
        return ResultadoOCR(texto="", linhas=[], bounding_boxes=[], confidences=[], confianca_media=0.0)

    import pytesseract
    from pytesseract import Output, TesseractError

    idioma_tesseract = _converter_idiomas(idiomas)
    imagem_tratada = _preparar_para_ocr(imagem)

    try:
        dados = pytesseract.image_to_data(
            imagem_tratada, lang=idioma_tesseract, output_type=Output.DICT,
            config="--psm 6",  # assume um bloco uniforme de texto — adequado a carimbos
        )
    except TesseractError as exc:
        # Código de retorno negativo = processo morto por sinal
        # (ex.: -2 é SIGINT, o Ctrl+C do usuário). Nesse caso não há
        # nada de errado com a instalação, e sugerir reinstalar o
        # pacote de idiomas só confunde quem acabou de cancelar.
        codigo = getattr(exc, "status", None)
        if isinstance(codigo, int) and codigo < 0:
            logger.info("OCR interrompido (processamento cancelado).")
        else:
            logger.error(
                "Erro do Tesseract (idioma '%s' instalado? tente 'brew install tesseract-lang'): %s",
                idioma_tesseract, exc,
            )
        return ResultadoOCR(texto="", linhas=[], bounding_boxes=[], confidences=[], confianca_media=0.0)

    linhas: list[str] = []
    boxes: list[list[list[float]]] = []
    confidences: list[float] = []

    total_palavras = len(dados.get("text", []))
    for indice in range(total_palavras):
        texto_palavra = dados["text"][indice].strip()
        confianca_bruta = dados["conf"][indice]
        if not texto_palavra or confianca_bruta in ("-1", -1):
            continue

        x, y, w, h = dados["left"][indice], dados["top"][indice], dados["width"][indice], dados["height"][indice]
        poligono = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

        linhas.append(texto_palavra)
        boxes.append(poligono)
        confidences.append(float(confianca_bruta) / 100.0)  # Tesseract usa 0-100; padronizamos para 0-1

    confianca_media = float(np.mean(confidences)) if confidences else 0.0
    texto_completo = " ".join(linhas)

    logger.debug("Tesseract extraiu %d palavras, confiança média=%.2f", len(linhas), confianca_media)

    return ResultadoOCR(
        texto=texto_completo,
        linhas=linhas,
        bounding_boxes=boxes,
        confidences=confidences,
        confianca_media=confianca_media,
    )
