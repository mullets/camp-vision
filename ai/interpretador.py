"""
ai/interpretador.py
====================
Interpretação, por IA, do texto bruto extraído do carimbo via OCR,
retornando metadados estruturados da prancha.

Usa a API da OpenAI (Chat Completions com saída JSON estruturada).
Caso a IA não esteja disponível (sem chave configurada, erro de rede,
ou desabilitada nas configurações), o sistema cai automaticamente
para um extrator baseado apenas em regras sobre o texto do OCR
(ver `ai/fallback_regras.py`), garantindo que o pipeline nunca pare
por falta de IA.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger("campvision.ai")

CAMPOS_METADADOS = [
    "projeto", "cliente", "arquiteto", "endereco", "cidade",
    "ano", "prancha", "numero", "escala", "fase", "tipo", "observacoes",
]

PROMPT_SISTEMA = """Você é um assistente especializado em interpretar textos extraídos \
por OCR de carimbos de pranchas de projetos de arquitetura brasileiros.

Dado o texto bruto do OCR (que pode conter ruído, erros de leitura e \
quebras de linha desordenadas), extraia os metadados da prancha e \
responda ESTRITAMENTE em JSON, sem nenhum texto adicional, usando \
exatamente estas chaves:

{"projeto": "", "cliente": "", "arquiteto": "", "endereco": "", \
"cidade": "", "ano": "", "prancha": "", "numero": "", "escala": "", \
"fase": "", "tipo": "", "observacoes": ""}

Se um campo não puder ser determinado com razoável confiança, deixe-o \
como string vazia. Não invente informação que não esteja implícita no \
texto."""


@dataclass
class MetadadosPrancha:
    projeto: str = ""
    cliente: str = ""
    arquiteto: str = ""
    endereco: str = ""
    cidade: str = ""
    ano: str = ""
    prancha: str = ""
    numero: str = ""
    escala: str = ""
    fase: str = ""
    tipo: str = ""
    observacoes: str = ""
    confianca_ia: float = 0.0
    fonte: str = "ia"  # "ia" ou "regras" (fallback)
    # Preenchidos depois, em scanner/lote._atribuir_codigos_por_projeto
    # (não pela interpretação por IA/regras) — deixados aqui como
    # campos explícitos da dataclass, em vez de atributos dinâmicos,
    # para o resto do código não depender de getattr(..., default).
    codigo_projeto_auto: str = ""
    sequencial_no_projeto: int = 0
    # Ano usado para a PASTA de arquivamento — um único valor por
    # projeto (o ano mais comum entre as pranchas do grupo), diferente
    # de `ano` (que continua sendo o ano lido/propagado individualmente
    # em cada prancha, usado no CSV e no EXIF). Sem isso, pranchas do
    # mesmo projeto com anos individualmente distintos (ou "Ano
    # desconhecido" só em algumas) fragmentariam um único projeto em
    # várias pastas de ano diferentes — ver
    # scanner/lote._atribuir_codigos_por_projeto.
    ano_pasta: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class InterpretadorIA:
    """Encapsula a chamada à API da OpenAI, com fallback automático
    para extração baseada em regras quando a IA não está disponível."""

    def __init__(self, api_key: str, modelo: str = "gpt-4o-mini", habilitada: bool = True):
        self.api_key = api_key
        self.modelo = modelo
        self.habilitada = habilitada
        self._client = None

    def _obter_cliente(self):
        if self._client is not None:
            return self._client
        from openai import OpenAI  # import tardio
        self._client = OpenAI(api_key=self.api_key)
        return self._client

    def disponivel(self) -> bool:
        return bool(self.habilitada and self.api_key)

    def interpretar(self, texto_ocr: str) -> MetadadosPrancha:
        """Interpreta o texto do OCR e retorna metadados estruturados.
        Cai automaticamente para o extrator por regras em caso de
        indisponibilidade ou erro da IA."""
        if not self.disponivel():
            logger.info("IA indisponível/desabilitada — usando extração por regras.")
            return self._fallback(texto_ocr)

        try:
            return self._interpretar_via_openai(texto_ocr)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falha ao consultar a IA (%s). Usando extração por regras.", exc)
            return self._fallback(texto_ocr)

    def _interpretar_via_openai(self, texto_ocr: str) -> MetadadosPrancha:
        cliente = self._obter_cliente()
        resposta = cliente.chat.completions.create(
            model=self.modelo,
            messages=[
                {"role": "system", "content": PROMPT_SISTEMA},
                {"role": "user", "content": texto_ocr},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        conteudo = resposta.choices[0].message.content
        dados = json.loads(conteudo)

        campos_validos = {k: str(v) if v is not None else "" for k, v in dados.items() if k in CAMPOS_METADADOS}
        metadados = MetadadosPrancha(**campos_validos)
        metadados.confianca_ia = 0.9  # a API não retorna confiança explícita; usamos um valor alto fixo
        metadados.fonte = "ia"
        return metadados

    def _fallback(self, texto_ocr: str) -> MetadadosPrancha:
        from ai.fallback_regras import extrair_por_regras
        return extrair_por_regras(texto_ocr)
