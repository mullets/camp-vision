"""
config.py
==========
Configuração central do CAMP Vision.

Centraliza todos os parâmetros configuráveis do sistema: caminhos,
credenciais, parâmetros de processamento de imagem, OCR, IA e
exportação.
"""

from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path

logger = logging.getLogger("campvision.config")

APP_NAME = "CAMP Vision"
APP_VERSION = "0.1.0"
# Identificador de build — muda a cada entrega.
VERSAO_BUILD = "2026-08-31-70-gpt"

USER_DIR = Path.home() / ".campvision"
CONFIG_PATH = USER_DIR / "config.json"
DB_PATH = USER_DIR / "campvision.sqlite3"
LOG_DIR = USER_DIR / "logs"

@dataclass
class Settings:
    idioma: str = "pt-BR"
    pasta_padrao: str = str(Path.home() / "Documents")
    tema: str = "escuro"
    miniatura_tamanho_px: int = 1500
    miniatura_qualidade: int = 90
    ocr_motor: str = "tesseract"
    ocr_modelo: str = "PP-OCRv4"
    ocr_idiomas: list = field(default_factory=lambda: ["pt", "en"])
    ia_modelo: str = "gpt-4o-mini"
    ia_api_key: str = ""
    ia_habilitada: bool = True
    quantidade_threads: int = max(1, (os.cpu_count() or 4) - 1)
    formatos_aceitos: list = field(default_factory=lambda: [".tif", ".tiff", ".jpg", ".jpeg", ".png", ".pdf"])
    deteccao_carimbo_modo: str = "heuristico"
    caminho_modelo_carimbo: str = ""
    carimbo_regiao_busca: str = "automatico"
    classificacao_modo: str = "regras"
    caminho_modelo_classificacao: str = ""
    confianca_minima_ml: float = 0.5
    tamanho_imagem_ml: int = 0
    deteccao_multiorientacao: bool = True
    renomeacao_habilitada: bool = True
    renomeacao_padrao: str = "{codigo_projeto_auto}-{sequencial_no_projeto} - {projeto} - {nome_prancha}"
    renomeacao_digitos_sequencial: int = 5
    gravar_metadados_exif: bool = True
    atribuicao_instituicao: str = "CAMP - Casa da Arquitetura Moderna Paulista"
    arquivamento_habilitado: bool = True
    arquivamento_padrao_pastas: str = "{ano}/{codigo_projeto_auto} - {projeto}"
    arquivamento_pasta_raiz: str = ""
    arquivamento_copiar: bool = True
    exportar_csv: bool = True
    exportar_xlsx: bool = True
    exportar_json: bool = True
    salvar_carimbos: bool = True
    salvar_miniaturas: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        campos_validos = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**campos_validos)

def garantir_diretorios() -> None:
    USER_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

_PADROES_RENOMEACAO_ANTIGOS = {
    "{codigo_projeto}-{prancha}-{sequencial}",
    "{codigo_projeto_auto}-{sequencial_no_projeto}",
}
_PADROES_ARQUIVAMENTO_ANTIGOS = {"{ano}/{projeto}"}

def _migrar(settings: Settings) -> Settings:
    alterado = False
    formatos_atuais = {f.lower() for f in settings.formatos_aceitos}
    faltando = [f for f in Settings().formatos_aceitos if f.lower() not in formatos_atuais]
    if faltando:
        settings.formatos_aceitos = list(settings.formatos_aceitos) + faltando
        alterado = True
    if settings.renomeacao_padrao in _PADROES_RENOMEACAO_ANTIGOS:
        settings.renomeacao_padrao = Settings().renomeacao_padrao
        alterado = True
    if settings.arquivamento_padrao_pastas in _PADROES_ARQUIVAMENTO_ANTIGOS:
        settings.arquivamento_padrao_pastas = Settings().arquivamento_padrao_pastas
        alterado = True
    if alterado:
        salvar_configuracoes(settings)
    return settings

def carregar_configuracoes() -> Settings:
    garantir_diretorios()
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return _migrar(Settings.from_dict(data))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Falha ao ler config.json (%s). Usando padrões.", exc)
            return Settings()
    settings = Settings()
    salvar_configuracoes(settings)
    return settings

def salvar_configuracoes(settings: Settings) -> None:
    garantir_diretorios()
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(settings.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info("Configurações salvas em %s", CONFIG_PATH)
