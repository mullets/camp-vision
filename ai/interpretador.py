"""
ai/interpretador.py
====================
Interpretação do texto OCR de carimbos/blocos de identificação de
pranchas arquitetônicas, retornando metadados estruturados.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger("campvision.ai")

CAMPOS_METADADOS = [
    "projeto", "cliente", "arquiteto", "endereco", "cidade",
    "ano", "prancha", "numero", "escala", "fase", "tipo", "observacoes",
]

PROMPT_SISTEMA = """Você é um assistente especializado em interpretar textos extraídos por OCR de carimbos e blocos de identificação de pranchas de projetos de arquitetura brasileiros.

O OCR pode ter erros, letras trocadas, acentos ausentes, quebras de linha e texto incompleto. Extraia somente o que estiver sustentado pelo texto. Não invente informação.

Responda ESTRITAMENTE em JSON, usando exatamente estas chaves:
{"projeto":"", "cliente":"", "arquiteto":"", "endereco":"", "cidade":"", "ano":"", "prancha":"", "numero":"", "escala":"", "fase":"", "tipo":"", "observacoes":""}

Para o campo tipo, reconheça títulos e expressões arquitetônicas mesmo que estejam no meio do texto. Exemplos: PLANTA, PLANTA PAVIMENTO TÉRREO, PLANTA BAIXA -> Planta; CORTE, CORTES, CORTE ESQUEMÁTICO -> Corte; FACHADA/FACHADAS -> Fachada; ELEVAÇÃO/ELEVAÇÕES/VISTA -> Elevação; IMPLANTAÇÃO/SITUAÇÃO/LOCAÇÃO -> Implantação; COBERTURA -> Cobertura; PERSPECTIVA -> Perspectiva; CROQUI/ESBOÇO -> Croqui; ESTRUTURA -> Estrutura; HIDRÁULICA -> Hidráulica; ELÉTRICA/ILUMINAÇÃO -> Elétrica; PAISAGISMO -> Paisagismo; ESQUADRIAS -> Esquadrias; ESCADA/ESCADAS -> Escadas; DETALHE/DETALHES/DETALHAMENTO -> Detalhes; MEMORIAL -> Memorial.

Se houver mais de um título, escolha o tipo principal da prancha. Não use palavras genéricas como "planta" se houver uma expressão mais específica que indique outro tipo. Se não houver evidência razoável, deixe tipo vazio.

O campo prancha deve conter o título/nome do desenho quando identificável. O campo numero deve conter apenas o número/código da folha quando identificável. Não confunda título com número."""

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
    fonte: str = "ia"
    codigo_projeto_auto: str = ""
    sequencial_no_projeto: int = 0
    ano_pasta: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

class InterpretadorIA:
    def __init__(self, api_key: str, modelo: str = "gpt-4o-mini", habilitada: bool = True):
        self.api_key = api_key
        self.modelo = modelo
        self.habilitada = habilitada
        self._client = None

    def _obter_cliente(self):
        if self._client is not None:
            return self._client
        from openai import OpenAI
        self._client = OpenAI(api_key=self.api_key)
        return self._client

    def disponivel(self) -> bool:
        return bool(self.habilitada and self.api_key)

    def interpretar(self, texto_ocr: str) -> MetadadosPrancha:
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
        metadados.confianca_ia = 0.9
        metadados.fonte = "ia"
        return metadados

    def _fallback(self, texto_ocr: str) -> MetadadosPrancha:
        from ai.fallback_regras import extrair_por_regras
        return extrair_por_regras(texto_ocr)
