"""
utils/texto.py
================
Normalização de texto compartilhada por módulos que precisam comparar
texto reconhecido por OCR contra palavras-chave de forma tolerante a
erros de reconhecimento (acentos perdidos, letras trocadas por
dígitos parecidos, etc.) — usada tanto na verificação de conteúdo do
carimbo (`scanner/detector_carimbo.py`) quanto na extração de
metadados por regras (`ai/fallback_regras.py`).
"""

from __future__ import annotations

import unicodedata


def normalizar_maiusculas(texto: str) -> str:
    """Remove acentos e converte para maiúsculas, para comparação
    tolerante a variações de acentuação entre o texto original e o
    que o OCR efetivamente reconheceu."""
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return texto.upper()
