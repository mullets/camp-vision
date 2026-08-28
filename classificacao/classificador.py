"""
classificacao/classificador.py
================================
Classificação automática do tipo de prancha arquitetônica.

Combina duas fontes de evidência:
  1. Palavras-chave encontradas no texto do OCR (título da prancha,
     texto do carimbo, legendas).
  2. Sugestão vinda da interpretação por IA (campo "tipo"), quando
     disponível — tratada como evidência de maior peso.

O resultado é sempre uma das categorias abaixo, ou "Não classificado"
quando não há evidência suficiente.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger("campvision.classificacao")

CATEGORIAS = [
    "Implantação", "Planta", "Cobertura", "Corte", "Fachada",
    "Elevação", "Perspectiva", "Croqui", "Estrutura", "Hidráulica",
    "Elétrica", "Paisagismo", "Esquadrias", "Escadas", "Detalhes",
    "Memorial",
]

# Palavras-chave (minúsculas) associadas a cada categoria. A ordem
# importa pouco aqui pois todas as categorias são avaliadas e a de
# maior número de ocorrências vence.
_PALAVRAS_CHAVE: dict[str, list[str]] = {
    "Implantação": ["implantação", "situação", "locação"],
    "Planta": ["planta baixa", "planta de situação", "plantas", "planta", "pavimento"],
    "Cobertura": ["cobertura", "telhado", "telha"],
    # Formas no plural são comuns nos carimbos reais ("CORTES e VISTA")
    # e precisam aparecer explicitamente: a busca é por substring, e
    # "corte " (com espaço) não casa com "cortes".
    "Corte": ["cortes", "corte aa", "corte bb", "corte transversal", "corte longitudinal", "corte "],
    "Fachada": ["fachadas", "fachada"],
    "Elevação": ["elevação", "elevações", "vista"],
    "Perspectiva": ["perspectiva", "3d", "render"],
    "Croqui": ["croqui", "esboço", "estudo preliminar"],
    "Estrutura": ["estrutura", "estrutural", "concreto armado", "fundação", "fundações", "laje", "pilar", "viga"],
    "Hidráulica": ["hidráulica", "hidrossanitário", "água fria", "água quente", "esgoto", "pluvial"],
    "Elétrica": ["elétrica", "elétrico", "iluminação", "tomadas", "quadro de cargas", "spda"],
    "Paisagismo": ["paisagismo", "paisagístico", "vegetação", "jardim"],
    "Esquadrias": ["esquadria", "esquadrias", "janela", "porta", "portas e janelas"],
    "Escadas": ["escada", "escadas", "rampa"],
    "Detalhes": ["detalhes", "detalhe", "detalhes construtivos", "detalhamento"],
    "Memorial": ["memorial descritivo", "memorial"],
}


@dataclass
class ResultadoClassificacao:
    tipo: str
    confianca: float
    evidencias: list[str]


def classificar(texto_ocr: str, tipo_sugerido_ia: str = "") -> ResultadoClassificacao:
    """Classifica o tipo da prancha combinando palavras-chave do OCR
    com a sugestão (opcional) vinda da interpretação por IA."""
    texto_lower = texto_ocr.lower()

    pontuacoes: dict[str, int] = {categoria: 0 for categoria in CATEGORIAS}
    evidencias: dict[str, list[str]] = {categoria: [] for categoria in CATEGORIAS}

    for categoria, palavras in _PALAVRAS_CHAVE.items():
        for palavra in palavras:
            ocorrencias = texto_lower.count(palavra)
            if ocorrencias:
                pontuacoes[categoria] += ocorrencias
                evidencias[categoria].append(palavra)

    # A sugestão da IA, se compatível com uma categoria conhecida,
    # recebe um peso extra considerável (equivalente a várias
    # ocorrências de palavra-chave no OCR).
    tipo_sugerido_normalizado = tipo_sugerido_ia.strip().title()
    if tipo_sugerido_normalizado in pontuacoes:
        pontuacoes[tipo_sugerido_normalizado] += 5
        evidencias[tipo_sugerido_normalizado].append(f"sugestão IA: {tipo_sugerido_ia}")

    melhor_categoria = max(pontuacoes, key=lambda c: pontuacoes[c])
    melhor_pontuacao = pontuacoes[melhor_categoria]

    if melhor_pontuacao == 0:
        logger.debug("Nenhuma evidência de classificação encontrada.")
        return ResultadoClassificacao(tipo="Não classificado", confianca=0.0, evidencias=[])

    total_pontos = sum(pontuacoes.values()) or 1
    confianca = min(melhor_pontuacao / total_pontos + 0.2, 1.0)

    return ResultadoClassificacao(
        tipo=melhor_categoria,
        confianca=round(confianca, 2),
        evidencias=evidencias[melhor_categoria],
    )


def _adaptador_regras(texto_ocr: str, tipo_sugerido_ia: str = "", imagem: Optional[np.ndarray] = None) -> ResultadoClassificacao:
    """Adapta `classificar` (baseada em texto) à assinatura comum usada
    pela fábrica de estratégia, que também aceita a imagem da prancha
    (usada apenas pelo modo por modelo treinado)."""
    return classificar(texto_ocr, tipo_sugerido_ia=tipo_sugerido_ia)


def criar_classificador(
    modo: str = "regras",
    caminho_modelo: Optional[str] = None,
) -> Callable[..., ResultadoClassificacao]:
    """Fábrica de estratégia: retorna a função de classificação de
    tipo de prancha a ser usada pelo pipeline.

    - "regras": usa `classificar` (palavras-chave do OCR + sugestão da
      IA), sempre disponível.
    - "modelo_treinado": carrega um classificador de imagem treinado
      (ver `ml/treinar_classificador.py`) que analisa a miniatura
      completa da prancha. Assim como no detector de carimbo, uma
      falha ao carregar o modelo treinado cai automaticamente para o
      classificador por regras, sem interromper o processamento.

    A função retornada aceita sempre `(texto_ocr, tipo_sugerido_ia="", imagem=None)`,
    para que o pipeline não precise saber qual estratégia está ativa.
    """
    if modo == "modelo_treinado":
        if not caminho_modelo:
            logger.warning("Modo 'modelo_treinado' selecionado sem caminho de modelo — usando classificação por regras.")
            return _adaptador_regras
        try:
            from ml.classificador_ml import ClassificadorML
            modelo = ClassificadorML(caminho_modelo)
            logger.info("Classificador de tipo por modelo treinado carregado: %s", caminho_modelo)

            def _adaptador_ml(texto_ocr: str, tipo_sugerido_ia: str = "", imagem: Optional[np.ndarray] = None) -> ResultadoClassificacao:
                if imagem is None:
                    logger.warning("Classificador treinado requer imagem, mas nenhuma foi fornecida — usando regras.")
                    return classificar(texto_ocr, tipo_sugerido_ia=tipo_sugerido_ia)
                return modelo.classificar(imagem)

            return _adaptador_ml
        except Exception as exc:  # noqa: BLE001
            logger.error("Falha ao carregar modelo treinado de classificação (%s). Usando regras.", exc)
            return _adaptador_regras

    return _adaptador_regras
