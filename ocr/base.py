"""
ocr/base.py
============
Estrutura de resultado compartilhada entre os motores de OCR
disponíveis (Tesseract, PaddleOCR), para que o restante do pipeline
não precise saber qual motor está em uso.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ResultadoOCR:
    texto: str
    linhas: list[str]
    bounding_boxes: list[list[list[float]]]  # lista de polígonos (4 pontos) por linha/palavra
    confidences: list[float]
    confianca_media: float
