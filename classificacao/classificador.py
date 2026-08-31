"""
Classificação automática do tipo de prancha arquitetônica.

Combina evidências do OCR e da interpretação da IA. O classificador
foi reforçado para títulos arquitetônicos curtos e variações comuns de
OCR, sem exigir que o texto inteiro seja perfeitamente reconhecido.
"""

from __future__ import annotations

import logging
import re
import unicodedata
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

_PALAVRAS_CHAVE: dict[str, list[str]] = {
    "Implantação": ["implantacao", "implantacao geral", "situacao", "locacao", "implant.", "implant"],
    "Planta": ["planta baixa", "planta de situacao", "planta", "plantas", "pavimento", "terreo", "subsolo", "mezanino"],
    "Cobertura": ["cobertura", "telhado", "telha", "planta de cobertura"],
    "Corte": ["cortes", "corte aa", "corte bb", "corte cc", "corte dd", "corte transversal", "corte longitudinal", "corte esquematico", "corte"],
    "Fachada": ["fachadas", "fachada", "fachada principal", "fachada lateral", "fachada posterior"],
    "Elevação": ["elevacao", "elevacoes", "vista", "vistas"],
    "Perspectiva": ["perspectiva", "perspectivas", "3d", "render", "renderizacao"],
    "Croqui": ["croqui", "croquis", "esboco", "estudo preliminar"],
    "Estrutura": ["estrutura", "estrutural", "concreto armado", "fundacao", "fundacoes", "laje", "pilar", "viga", "armacao"],
    "Hidráulica": ["hidraulica", "hidrossanitario", "agua fria", "agua quente", "esgoto", "pluvial"],
    "Elétrica": ["eletrica", "eletrico", "iluminacao", "tomadas", "quadro de cargas", "spda"],
    "Paisagismo": ["paisagismo", "paisagistico", "vegetacao", "jardim"],
    "Esquadrias": ["esquadria", "esquadrias", "janela", "porta", "portas e janelas"],
    "Escadas": ["escada", "escadas", "rampa"],
    "Detalhes": ["detalhes", "detalhe", "detalhes construtivos", "detalhamento", "detalhamento construtivo"],
    "Memorial": ["memorial descritivo", "memorial"],
}

# Frases de título são evidência muito mais forte que palavras soltas.
_PHRASES_FORTES = {
    "Implantação": ["planta de implantacao", "planta de situacao", "implantacao geral"],
    "Planta": ["planta pavimento", "planta do pavimento", "planta baixa", "planta terreo", "planta superior", "planta subsolo"],
    "Cobertura": ["planta de cobertura", "planta cobertura"],
    "Corte": ["corte esquematico", "corte transversal", "corte longitudinal", "cortes e vista", "cortes e vistas"],
    "Fachada": ["fachada principal", "fachada lateral", "fachada posterior", "fachadas"],
    "Elevação": ["elevacao principal", "elevacoes", "vista frontal", "vista lateral"],
    "Perspectiva": ["perspectiva geral", "perspectiva externa", "perspectiva interna"],
    "Croqui": ["estudo preliminar", "croqui geral"],
    "Detalhes": ["detalhes construtivos", "detalhamento construtivo"],
}

_ALIASES_IA = {
    "implantacao": "Implantação", "implantação": "Implantação",
    "planta": "Planta", "planta baixa": "Planta",
    "cobertura": "Cobertura", "corte": "Corte", "cortes": "Corte",
    "fachada": "Fachada", "fachadas": "Fachada",
    "elevacao": "Elevação", "elevação": "Elevação", "vista": "Elevação",
    "perspectiva": "Perspectiva", "croqui": "Croqui", "estrutura": "Estrutura",
    "hidraulica": "Hidráulica", "hidráulica": "Hidráulica",
    "eletrica": "Elétrica", "elétrica": "Elétrica", "paisagismo": "Paisagismo",
    "esquadrias": "Esquadrias", "escadas": "Escadas", "detalhes": "Detalhes",
    "memorial": "Memorial",
}

@dataclass
class ResultadoClassificacao:
    tipo: str
    confianca: float
    evidencias: list[str]


def _sem_acentos(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texto.lower())
        if unicodedata.category(c) != "Mn"
    )


def _normalizar_tipo_ia(valor: str) -> str:
    bruto = _sem_acentos(valor.strip())
    if bruto in _ALIASES_IA:
        return _ALIASES_IA[bruto]
    # Aceita respostas como "Planta arquitetônica" sem tornar toda
    # resposta livre da IA uma categoria válida.
    for alias, categoria in _ALIASES_IA.items():
        if len(alias) >= 5 and re.search(rf"\b{re.escape(alias)}\b", bruto):
            return categoria
    return ""


def classificar(texto_ocr: str, tipo_sugerido_ia: str = "") -> ResultadoClassificacao:
    """Classifica usando títulos/frases fortes, palavras-chave e IA.

    Frases de título recebem peso alto. Isso permite reconhecer, por
    exemplo, "PLANTA PAVIMENTO TERREO" mesmo quando o restante do
    carimbo é ilegível. A IA só recebe peso máximo quando sua resposta
    pode ser normalizada para uma categoria conhecida.
    """
    texto = _sem_acentos(texto_ocr)
    # Preserva espaços e reduz ruído típico do OCR para facilitar as
    # buscas de frases sem alterar o texto usado nas evidências.
    texto = re.sub(r"\s+", " ", texto).strip()

    pontuacoes: dict[str, float] = {categoria: 0.0 for categoria in CATEGORIAS}
    evidencias: dict[str, list[str]] = {categoria: [] for categoria in CATEGORIAS}

    for categoria, frases in _PHRASES_FORTES.items():
        for frase in frases:
            ocorrencias = texto.count(frase)
            if ocorrencias:
                pontuacoes[categoria] += 5.0 * ocorrencias
                evidencias[categoria].append(f"título: {frase}")

    for categoria, palavras in _PALAVRAS_CHAVE.items():
        for palavra in palavras:
            p = _sem_acentos(palavra)
            # Termos muito curtos exigem fronteira de palavra para não
            # gerar falsos acertos em palavras maiores.
            if len(p) <= 5:
                ocorrencias = len(re.findall(rf"\b{re.escape(p)}\b", texto))
            else:
                ocorrencias = texto.count(p)
            if ocorrencias:
                pontuacoes[categoria] += 1.0 * ocorrencias
                evidencias[categoria].append(palavra)

    tipo_ia = _normalizar_tipo_ia(tipo_sugerido_ia)
    if tipo_ia:
        pontuacoes[tipo_ia] += 6.0
        evidencias[tipo_ia].append(f"sugestão IA: {tipo_sugerido_ia}")

    melhor_categoria = max(pontuacoes, key=pontuacoes.get)
    melhor_pontuacao = pontuacoes[melhor_categoria]
    if melhor_pontuacao <= 0:
        logger.debug("Nenhuma evidência de classificação encontrada.")
        return ResultadoClassificacao(tipo="Não classificado", confianca=0.0, evidencias=[])

    total = sum(pontuacoes.values()) or 1.0
    # A confiança representa a dominância da melhor hipótese, com um
    # pequeno piso quando há uma frase de título explícita.
    confianca = melhor_pontuacao / total
    if any(ev.startswith("título:") for ev in evidencias[melhor_categoria]):
        confianca = max(confianca, 0.75)
    if tipo_ia == melhor_categoria:
        confianca = max(confianca, 0.80)

    return ResultadoClassificacao(
        tipo=melhor_categoria,
        confianca=round(min(confianca, 1.0), 2),
        evidencias=evidencias[melhor_categoria],
    )


def _adaptador_regras(texto_ocr: str, tipo_sugerido_ia: str = "", imagem: Optional[np.ndarray] = None) -> ResultadoClassificacao:
    return classificar(texto_ocr, tipo_sugerido_ia=tipo_sugerido_ia)


def criar_classificador(
    modo: str = "regras",
    caminho_modelo: Optional[str] = None,
) -> Callable[..., ResultadoClassificacao]:
    if modo == "modelo_treinado":
        if not caminho_modelo:
            logger.warning("Modo 'modelo_treinado' sem caminho — usando regras.")
            return _adaptador_regras
        try:
            from ml.classificador_ml import ClassificadorML
            modelo = ClassificadorML(caminho_modelo)
            logger.info("Classificador de tipo por modelo treinado carregado: %s", caminho_modelo)

            def _adaptador_ml(texto_ocr: str, tipo_sugerido_ia: str = "", imagem: Optional[np.ndarray] = None) -> ResultadoClassificacao:
                if imagem is None:
                    return classificar(texto_ocr, tipo_sugerido_ia=tipo_sugerido_ia)
                return modelo.classificar(imagem)

            return _adaptador_ml
        except Exception as exc:  # noqa: BLE001
            logger.error("Falha ao carregar modelo treinado (%s). Usando regras.", exc)
            return _adaptador_regras
    return _adaptador_regras
