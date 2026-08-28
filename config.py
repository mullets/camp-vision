"""
config.py
==========
Configuração central do CAMP Vision.

Centraliza todos os parâmetros configuráveis do sistema: caminhos,
credenciais, parâmetros de processamento de imagem, OCR, IA e
exportação. As configurações são persistidas em um arquivo JSON no
diretório de configuração do usuário (~/.campvision/config.json) e
podem ser editadas pela tela de Configurações da interface gráfica.
"""

from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("campvision.config")

APP_NAME = "CAMP Vision"
APP_VERSION = "0.1.0"
# Identificador de build — muda a cada vez que uma correção é entregue,
# para o usuário confirmar visualmente (terminal + título da janela)
# que está rodando a versão mais recente e não uma pasta antiga.
VERSAO_BUILD = "2026-08-14-69"

# Diretório de configuração e dados do usuário
USER_DIR = Path.home() / ".campvision"
CONFIG_PATH = USER_DIR / "config.json"
DB_PATH = USER_DIR / "campvision.sqlite3"
LOG_DIR = USER_DIR / "logs"


@dataclass
class Settings:
    """Configurações persistentes do CAMP Vision."""

    # Geral
    idioma: str = "pt-BR"
    pasta_padrao: str = str(Path.home() / "Documents")
    tema: str = "escuro"  # "claro" ou "escuro"

    # Miniaturas
    miniatura_tamanho_px: int = 1500
    miniatura_qualidade: int = 90

    # OCR
    ocr_motor: str = "tesseract"  # "tesseract" (padrão, leve) ou "paddleocr" (requer AVX/AVX2)
    ocr_modelo: str = "PP-OCRv4"  # relevante apenas quando ocr_motor="paddleocr"
    ocr_idiomas: list = field(default_factory=lambda: ["pt", "en"])

    # IA
    ia_modelo: str = "gpt-4o-mini"
    ia_api_key: str = ""
    ia_habilitada: bool = True

    # Processamento
    quantidade_threads: int = max(1, (os.cpu_count() or 4) - 1)
    formatos_aceitos: list = field(default_factory=lambda: [".tif", ".tiff", ".jpg", ".jpeg", ".png", ".pdf"])

    # Detecção de carimbo: "heuristico" (contornos/OpenCV) ou "modelo_treinado" (YOLO)
    deteccao_carimbo_modo: str = "heuristico"
    caminho_modelo_carimbo: str = ""
    # Região de busca do carimbo: "automatico" (4 cantos + faixas de borda)
    # ou uma região fixa (ex. "inferior_direito", "faixa_direita") —
    # recomendado fixar quando se sabe onde o carimbo fica no projeto,
    # pois reduz falsos positivos em desenhos técnicos cheios de linhas.
    carimbo_regiao_busca: str = "automatico"

    # Classificação de tipo de prancha: "regras" (palavras-chave) ou "modelo_treinado" (ResNet18)
    classificacao_modo: str = "regras"
    caminho_modelo_classificacao: str = ""

    # Confiança mínima para aceitar uma predição de modelo treinado
    confianca_minima_ml: float = 0.5
    # Tamanho de imagem usado na inferência do modelo treinado — 0
    # (padrão) detecta automaticamente a partir do próprio arquivo do
    # modelo; só é preciso informar manualmente em casos excepcionais.
    tamanho_imagem_ml: int = 0
    # Procura o carimbo em todas as orientações da prancha (girada e
    # espelhada) e usa a de maior confiança. Detectores treinados não
    # são invariantes a rotação, e as pranchas foram digitalizadas em
    # orientações variadas — sem isto, o modelo recebe a prancha numa
    # orientação diferente da que viu no treino e a confiança despenca.
    deteccao_multiorientacao: bool = True

    # Renomeação dos arquivos originais com código único de arquivamento
    renomeacao_habilitada: bool = True
    # Código automático do projeto (prefixo do arquiteto/cidade + P0001,
    # ex. "OCG-P0001") + posição da prancha no projeto (N0012) + nome
    # do projeto + título da prancha lido do carimbo (ou tipo+número,
    # se o carimbo não trouxer título legível). Ex.:
    # "OCG-P0003-N0012 - TEATRO DE SANTOS - PLANTA 01"
    renomeacao_padrao: str = "{codigo_projeto_auto}-{sequencial_no_projeto} - {projeto} - {nome_prancha}"
    renomeacao_digitos_sequencial: int = 5

    # Gravação de metadados (EXIF/IPTC/XMP) no arquivo final, via exiftool
    gravar_metadados_exif: bool = True
    # Sufixo de atribuição institucional gravado no campo Copyright do
    # EXIF, junto com o nome do arquiteto (formato:
    # "{arquiteto} / {atribuicao_instituicao}") — em vez do nome do
    # cliente, que é o que ficava lá antes por padrão.
    atribuicao_instituicao: str = "CAMP - Casa da Arquitetura Moderna Paulista"

    # Organização física em pastas (ano/projeto), alinhada às diretrizes
    # da Resolução CONARQ/MGI nº 56/2024 para acervos de arquitetura
    arquivamento_habilitado: bool = True
    # Pasta por ano e, dentro dele, por PROJETO (código automático +
    # nome), não pelo nome da pasta selecionada — um lote com vários
    # projetos diferentes gera uma pasta por projeto.
    arquivamento_padrao_pastas: str = "{ano}/{codigo_projeto_auto} - {projeto}"
    arquivamento_pasta_raiz: str = ""  # vazio = pasta irmã "<pasta selecionada>_catalogado"
    # Por padrão COPIA os arquivos originais renomeados pra pasta de
    # arquivamento (ano/projeto), em vez de mover — o original fica
    # intacto no lugar de origem. Só muda pra False se quiser
    # economizar espaço em disco e já confiar no fluxo.
    arquivamento_copiar: bool = True

    # Exportação
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
    """Garante que os diretórios de dados do usuário existam."""
    USER_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


# Padrões default de versões anteriores — usados só para decidir se um
# config.json salvo ainda está no valor padrão antigo (e pode ser
# migrado com segurança) ou se o usuário customizou de propósito (e
# nesse caso NUNCA sobrescrevemos a escolha dele).
_PADROES_RENOMEACAO_ANTIGOS = {
    "{codigo_projeto}-{prancha}-{sequencial}",
    "{codigo_projeto_auto}-{sequencial_no_projeto}",
}
_PADROES_ARQUIVAMENTO_ANTIGOS = {
    "{ano}/{projeto}",
}


def _migrar(settings: Settings) -> Settings:
    """Ajusta uma configuração salva em versão anterior do app.

    Sem isto, um recurso novo fica invisível para quem já usa o
    programa: a lista de formatos aceitos é PERSISTIDA, então quando
    o suporte a PDF foi adicionado, quem já tinha um config.json
    continuou com a lista antiga e via "0 arquivos encontrados" numa
    pasta cheia de PDFs. O usuário não tem como adivinhar que precisa
    editar a configuração para ganhar um formato novo. O mesmo vale
    para os padrões de nomenclatura/pastas: se ainda estiverem no
    valor default de uma versão anterior (nunca editados à mão), são
    atualizados para o novo default (organização por código de
    projeto); se o usuário já customizou, o valor dele é respeitado."""
    alterado = False

    formatos_atuais = {f.lower() for f in settings.formatos_aceitos}
    faltando = [f for f in Settings().formatos_aceitos if f.lower() not in formatos_atuais]
    if faltando:
        settings.formatos_aceitos = list(settings.formatos_aceitos) + faltando
        logger.info("Formatos de arquivo recém-suportados adicionados à configuração: %s",
                    ", ".join(faltando))
        alterado = True

    if settings.renomeacao_padrao in _PADROES_RENOMEACAO_ANTIGOS:
        settings.renomeacao_padrao = Settings().renomeacao_padrao
        logger.info("Padrão de nomenclatura desatualizado — atualizado para organizar por "
                    "código de projeto e nome da prancha: %s", settings.renomeacao_padrao)
        alterado = True

    if settings.arquivamento_padrao_pastas in _PADROES_ARQUIVAMENTO_ANTIGOS:
        settings.arquivamento_padrao_pastas = Settings().arquivamento_padrao_pastas
        logger.info("Padrão de pastas de arquivamento desatualizado — atualizado para organizar "
                    "por código de projeto: %s", settings.arquivamento_padrao_pastas)
        alterado = True

    if alterado:
        salvar_configuracoes(settings)
    return settings


def carregar_configuracoes() -> Settings:
    """Carrega as configurações do disco, criando um arquivo padrão se necessário."""
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
    """Persiste as configurações no disco em formato JSON."""
    garantir_diretorios()
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(settings.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info("Configurações salvas em %s", CONFIG_PATH)


# Instância global de configurações, carregada uma única vez no import.
settings: Settings = carregar_configuracoes()
