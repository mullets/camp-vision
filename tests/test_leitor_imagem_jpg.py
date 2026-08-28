"""Testes de scanner.leitor_imagem para arquivos de imagem simples
(JPG/PNG) — em especial o aviso de resolução baixa quando o arquivo
não tem DPI gravado, comum em JPG (ao contrário de TIFF, onde a tag
de resolução quase sempre está presente)."""

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scanner.leitor_imagem import carregar_paginas


def test_jpg_colorido_carrega_com_cor_preservada(tmp_path):
    caminho = tmp_path / "colorida.jpg"
    imagem = np.zeros((500, 500, 3), dtype=np.uint8)
    imagem[:, :, 2] = 200  # canal R bem mais forte que os outros (BGR)
    cv2.imwrite(str(caminho), imagem, [cv2.IMWRITE_JPEG_QUALITY, 95])

    paginas = list(carregar_paginas(caminho))
    assert len(paginas) == 1
    media = paginas[0].imagem.mean(axis=(0, 1))
    assert media[2] > media[0] + 50  # canal R nitidamente mais forte que B


def test_jpg_sem_dpi_e_pequeno_gera_aviso(tmp_path, caplog):
    caminho = tmp_path / "pequena.jpg"
    imagem = np.full((800, 1000, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(caminho), imagem)  # cv2.imwrite não grava DPI

    with caplog.at_level(logging.WARNING):
        list(carregar_paginas(caminho))

    assert any("Resolução possivelmente baixa" in r.message for r in caplog.records)


def test_jpg_sem_dpi_mas_grande_nao_gera_aviso(tmp_path, caplog):
    caminho = tmp_path / "grande.jpg"
    imagem = np.full((3000, 4000, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(caminho), imagem)

    with caplog.at_level(logging.WARNING):
        list(carregar_paginas(caminho))

    assert not any("Resolução" in r.message for r in caplog.records)


def test_jpg_com_dpi_baixo_gravado_usa_o_dpi_nao_o_piso_de_pixels(tmp_path, caplog):
    caminho = tmp_path / "dpi_baixo.jpg"
    Image.new("RGB", (3000, 4000), "white").save(str(caminho), dpi=(100, 100))

    with caplog.at_level(logging.WARNING):
        list(carregar_paginas(caminho))

    mensagens = [r.message for r in caplog.records]
    assert any("Resolução baixa (100" in m for m in mensagens)
    assert not any("possivelmente baixa" in m for m in mensagens)


def test_jpg_com_dpi_bom_gravado_nao_gera_aviso(tmp_path, caplog):
    caminho = tmp_path / "dpi_bom.jpg"
    Image.new("RGB", (3000, 4000), "white").save(str(caminho), dpi=(300, 300))

    with caplog.at_level(logging.WARNING):
        list(carregar_paginas(caminho))

    assert not any("Resolução" in r.message for r in caplog.records)
