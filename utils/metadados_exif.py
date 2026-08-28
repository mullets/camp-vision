"""
utils/metadados_exif.py
=========================
Grava os metadados extraídos de cada prancha diretamente no arquivo
TIFF final (EXIF + IPTC + XMP), para que o acervo fique pesquisável
por ferramentas de indexação de metadados — Spotlight no macOS, Adobe
Bridge, Photo Mechanic, DAMs em geral — sem depender de abrir o CSV.

A gravação é feita via `exiftool` (Phil Harvey's ExifTool), a
ferramenta padrão de fato para escrever metadados em arquivos de
imagem, incluindo TIFF (o próprio formato EXIF é, historicamente, um
subconjunto de tags TIFF — por isso TIFFs suportam esses campos
nativamente). O ExifTool não é uma dependência Python; precisa estar
instalado no sistema:

    brew install exiftool

Se o `exiftool` não estiver disponível, a gravação de metadados é
simplesmente pulada (com um aviso no log) — isso nunca interrompe o
processamento do lote.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("campvision.metadados_exif")

_CAMINHO_EXIFTOOL: Optional[str] = None
_VERIFICADO = False


@dataclass
class MetadadosParaGravar:
    """Metadados de uma prancha a serem escritos no arquivo final."""

    projeto: str = ""
    cliente: str = ""
    arquiteto: str = ""
    endereco: str = ""
    cidade: str = ""
    ano: str = ""
    prancha: str = ""
    numero: str = ""
    escala: str = ""
    tipo: str = ""
    fase: str = ""
    observacoes: str = ""
    codigo_gerado: str = ""
    # Sufixo institucional pro campo Copyright — ver config.atribuicao_instituicao
    atribuicao_instituicao: str = ""


def exiftool_disponivel() -> bool:
    """Verifica (uma única vez, com cache) se o binário `exiftool`
    está disponível no PATH do sistema."""
    global _CAMINHO_EXIFTOOL, _VERIFICADO
    if not _VERIFICADO:
        _CAMINHO_EXIFTOOL = shutil.which("exiftool")
        _VERIFICADO = True
        if _CAMINHO_EXIFTOOL is None:
            logger.warning(
                "exiftool não encontrado no PATH — metadados EXIF/IPTC não serão gravados nos arquivos. "
                "Instale com 'brew install exiftool' para habilitar esse recurso."
            )
    return _CAMINHO_EXIFTOOL is not None


def _montar_palavras_chave(m: MetadadosParaGravar) -> list[str]:
    """Monta a lista de palavras-chave (keywords) usada para busca,
    combinando os campos mais úteis para localizar a prancha depois."""
    candidatos = [m.tipo, m.cidade, m.ano, m.fase, m.projeto, m.cliente, m.arquiteto, m.numero]
    return [c.strip() for c in candidatos if c and c.strip()]


def _montar_copyright(m: MetadadosParaGravar) -> str:
    """Copyright = arquiteto + atribuição institucional (ex.: "Fulano
    de Tal / CAMP - Casa da Arquitetura Moderna Paulista"), em vez do
    nome do cliente — que é o dono/contratante da obra, não quem tem
    direitos sobre o desenho em si."""
    partes = [p for p in [m.arquiteto, m.atribuicao_instituicao] if p and p.strip()]
    return " / ".join(partes)


def _montar_descricao(m: MetadadosParaGravar) -> str:
    partes = [p for p in [m.projeto, m.prancha, m.tipo] if p]
    return " — ".join(partes) if partes else ""


def gravar_metadados(caminho_arquivo: Path, metadados: MetadadosParaGravar) -> bool:
    """Grava os metadados da prancha no arquivo (EXIF/IPTC/XMP) usando
    exiftool. Retorna True se a gravação foi bem-sucedida, False se
    foi pulada ou falhou (nunca lança exceção — falhas de metadado não
    devem interromper o processamento do lote)."""
    if not exiftool_disponivel():
        return False

    palavras_chave = _montar_palavras_chave(metadados)
    descricao = _montar_descricao(metadados)
    copyright_texto = _montar_copyright(metadados)

    argumentos = [
        _CAMINHO_EXIFTOOL,
        "-overwrite_original",
        "-codedcharacterset=utf8",
        f"-ImageDescription={descricao}",
        f"-XMP-dc:Description={descricao}",
        f"-XMP-dc:Title={metadados.numero or metadados.prancha}",
        f"-Artist={metadados.arquiteto}",
        f"-XMP-dc:Creator={metadados.arquiteto}",
        f"-Copyright={copyright_texto}",
        f"-Software=CAMP Vision",
        f"-XMP-dc:Subject={';'.join(palavras_chave)}" if palavras_chave else None,
        f"-IPTC:Keywords={','.join(palavras_chave)}" if palavras_chave else None,
        f"-XMP-photoshop:City={metadados.cidade}" if metadados.cidade else None,
        f"-XMP-dc:Rights={copyright_texto}" if copyright_texto else None,
    ]
    # Remove argumentos vazios/None (campos sem valor não são escritos)
    argumentos = [a for a in argumentos if a is not None and not a.endswith("=")]
    argumentos.append(str(caminho_arquivo))

    try:
        resultado = subprocess.run(
            argumentos, capture_output=True, text=True, timeout=30, check=False,
        )
        if resultado.returncode != 0:
            logger.warning("exiftool retornou erro para %s: %s", caminho_arquivo.name, resultado.stderr.strip())
            return False
        logger.debug("Metadados gravados em %s", caminho_arquivo.name)
        return True
    except subprocess.TimeoutExpired:
        logger.warning("Timeout ao gravar metadados em %s", caminho_arquivo.name)
        return False
    except OSError as exc:
        logger.warning("Falha ao executar exiftool para %s: %s", caminho_arquivo.name, exc)
        return False
