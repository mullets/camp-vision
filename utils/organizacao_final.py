"""
utils/organizacao_final.py
===========================
Organização final de projetos processados pelo modo automatizado.

O Windows pode deixar um projeto em uma pasta de saída temporária (por
exemplo, "99 - Saida Scanner Contex HD"). Depois da catalogação, o CAMP
Vision usa os metadados finais para colocar o projeto no fundo correto e
normalizar a estrutura interna do projeto.

Regras importantes:
- preserva o nome da pasta do projeto já definido pelo padrão CAMP;
- aproveita um fundo existente pelo código (ex.: F002);
- se o fundo ainda não existir, cria "<código> - <arquiteto>" usando o
  arquiteto catalogado/canonizado pelo banco de conhecimento;
- nunca sobrescreve um projeto já existente;
- reorganiza TIF/JPG que estejam soltos na raiz para a estrutura CAMP
  "01 - Desenhos e Pranchas/...";
- move a pasta inteira em vez de copiar os TIFFs novamente.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Optional

from database.models import Arquiteto, criar_sessao

logger = logging.getLogger("campvision.organizacao")

SERIE_PADRAO = "01 - Desenhos e Pranchas"
TIPO_PREVIEW = "03 - Preview (JPG)"
TIPO_ARQUIVISTICO = "01 - Arquivo Arquivístico (TIFF)"
PADRAO_FUNDO = re.compile(r"^(F\d{3,})\s*-\s*(.+)$", re.IGNORECASE)
CODIGO_FUNDO = re.compile(r"\b(F\d{3,})\b", re.IGNORECASE)


def _normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto or "")
    return "".join(c for c in texto if not unicodedata.combining(c)).upper().strip()


def _ler_json(caminho: Path) -> dict:
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        return dados if isinstance(dados, dict) else {}
    except Exception as exc:
        logger.warning("[organização] Não consegui ler %s: %s", caminho, exc)
        return {}


def _valores_recursivos(obj) -> list[str]:
    valores: list[str] = []
    if isinstance(obj, dict):
        for valor in obj.values():
            valores.extend(_valores_recursivos(valor))
    elif isinstance(obj, list):
        for valor in obj:
            valores.extend(_valores_recursivos(valor))
    elif isinstance(obj, str):
        valores.append(obj.strip())
    return valores


def _extrair_codigo_fundo(pasta_projeto: Path, info: dict) -> Optional[str]:
    """Procura primeiro no nome do fundo atual, depois no projeto/info."""
    for texto in (pasta_projeto.parent.name, pasta_projeto.name):
        match = CODIGO_FUNDO.search(texto)
        if match:
            return match.group(1).upper()

    chaves_prioritarias = (
        "codigo_fundo", "fundo_codigo", "codigoFundo", "fundo", "codigo_acervo",
    )
    for chave in chaves_prioritarias:
        valor = info.get(chave)
        if isinstance(valor, str):
            match = CODIGO_FUNDO.search(valor)
            if match:
                return match.group(1).upper()
        elif isinstance(valor, dict):
            for item in _valores_recursivos(valor):
                match = CODIGO_FUNDO.search(item)
                if match:
                    return match.group(1).upper()

    for valor in _valores_recursivos(info):
        match = CODIGO_FUNDO.search(valor)
        if match:
            return match.group(1).upper()
    return None


def _arquiteto_dominante(catalogacao: list[dict]) -> str:
    valores = [str(reg.get("Arquiteto", "")).strip() for reg in catalogacao]
    valores = [v for v in valores if v]
    if not valores:
        return ""

    grupos: dict[str, Counter] = {}
    for valor in valores:
        grupos.setdefault(_normalizar(valor), Counter())[valor] += 1
    grupo = max(grupos.values(), key=sum)
    return grupo.most_common(1)[0][0]


def _canonizar_arquiteto(arquiteto: str, db_path: Optional[str]) -> str:
    """Usa a tabela de arquitetos como fonte de grafia canônica quando possível."""
    if not arquiteto or not db_path:
        return arquiteto
    sessao = None
    try:
        sessao = criar_sessao(db_path)
        chave = _normalizar(arquiteto)
        candidato = sessao.query(Arquiteto).filter(Arquiteto.nome_normalizado == chave).first()
        if candidato and candidato.nome:
            return candidato.nome
    except Exception as exc:
        logger.debug("[organização] Banco de arquitetos indisponível como fallback: %s", exc)
    finally:
        if sessao is not None:
            sessao.close()
    return arquiteto


def _encontrar_raiz_arquivos(qnap_path: Path) -> Path:
    """Encontra o diretório 'Arquivos' na cadeia do QNAP.

    Isso permite que o mesmo código funcione tanto com
    /.../Arquivos/99 - Saida Scanner Contex HD quanto com uma pasta de
    entrada configurada diretamente em /.../Arquivos.
    """
    for ancestral in (qnap_path, *qnap_path.parents):
        if ancestral.name.strip().lower() == "arquivos":
            return ancestral
    return qnap_path.parent


def _encontrar_ou_criar_fundo(
    raiz_arquivos: Path,
    codigo_fundo: str,
    arquiteto: str,
) -> Optional[Path]:
    candidatos = []
    try:
        for item in raiz_arquivos.iterdir():
            if not item.is_dir():
                continue
            match = PADRAO_FUNDO.match(item.name.strip())
            if match and match.group(1).upper() == codigo_fundo.upper():
                candidatos.append(item)
    except OSError as exc:
        logger.error("[organização] Não consegui listar a raiz %s: %s", raiz_arquivos, exc)
        return None

    if candidatos:
        candidatos.sort(key=lambda p: p.name.lower())
        return candidatos[0]

    if not arquiteto:
        logger.warning(
            "[organização] Fundo %s não existe e o arquiteto não foi identificado; "
            "não vou inventar o nome da pasta.",
            codigo_fundo,
        )
        return None

    nome = f"{codigo_fundo} - {arquiteto}".strip()
    destino = raiz_arquivos / nome
    try:
        destino.mkdir(parents=True, exist_ok=True)
        logger.info("[organização] Fundo novo criado: %s", destino)
        return destino
    except OSError as exc:
        logger.error("[organização] Falha ao criar fundo %s: %s", destino, exc)
        return None


def _mover_conteudo_sem_sobrescrever(origem: Path, destino: Path) -> bool:
    destino.mkdir(parents=True, exist_ok=True)
    ok = True
    for item in sorted(origem.iterdir()):
        alvo = destino / item.name
        if alvo.exists():
            logger.warning("[organização] Não sobrescrevendo %s; deixado em %s.", alvo, item)
            ok = False
            continue
        shutil.move(str(item), str(alvo))
    try:
        origem.rmdir()
    except OSError:
        pass
    return ok


def _normalizar_subpastas_projeto(pasta_projeto: Path) -> bool:
    """Converte TIF/JPG soltos para a estrutura padrão CAMP, sem sobrescrever."""
    serie = pasta_projeto / SERIE_PADRAO
    serie.mkdir(parents=True, exist_ok=True)
    ok = True

    aliases = {
        "TIF": serie / TIPO_ARQUIVISTICO,
        "TIFF": serie / TIPO_ARQUIVISTICO,
        "JPG": serie / TIPO_PREVIEW,
        "JPEG": serie / TIPO_PREVIEW,
    }
    for nome, destino in aliases.items():
        origem = pasta_projeto / nome
        if not origem.is_dir() or origem.resolve() == destino.resolve():
            continue
        if not _mover_conteudo_sem_sobrescrever(origem, destino):
            ok = False
        logger.info("[organização] Pasta %s normalizada para %s", origem, destino)

    # Alguns lotes podem chegar com as pastas finais já criadas na raiz.
    for nome in (TIPO_ARQUIVISTICO, TIPO_PREVIEW):
        origem = pasta_projeto / nome
        destino = serie / nome
        if origem.is_dir() and origem.resolve() != destino.resolve():
            if not _mover_conteudo_sem_sobrescrever(origem, destino):
                ok = False
            logger.info("[organização] Pasta %s movida para %s", origem, destino)

    return ok


def organizar_projeto_catalogado(
    pasta_projeto: Path,
    qnap_acervos_path: Path,
    db_path: Optional[str] = None,
) -> Optional[Path]:
    """Normaliza e move um projeto catalogado para o fundo correto.

    Retorna o novo caminho quando a organização termina com sucesso;
    retorna None quando não é seguro decidir o destino.
    """
    catalogacao_path = pasta_projeto / "catalogacao" / "catalogacao.json"
    catalogacao: list[dict] = []
    if catalogacao_path.exists():
        try:
            dados = json.loads(catalogacao_path.read_text(encoding="utf-8"))
            if isinstance(dados, list):
                catalogacao = dados
        except Exception as exc:
            logger.warning("[organização] catalogacao.json inválido em %s: %s", catalogacao_path, exc)

    info = _ler_json(pasta_projeto / "info_projeto.json") if (pasta_projeto / "info_projeto.json").exists() else {}
    codigo_fundo = _extrair_codigo_fundo(pasta_projeto, info)
    if not codigo_fundo:
        logger.warning("[organização] Não identifiquei código de fundo para %s.", pasta_projeto.name)
        return None

    arquiteto = _arquiteto_dominante(catalogacao)
    arquiteto = _canonizar_arquiteto(arquiteto, db_path)

    raiz_arquivos = _encontrar_raiz_arquivos(qnap_acervos_path)
    fundo = _encontrar_ou_criar_fundo(raiz_arquivos, codigo_fundo, arquiteto)
    if fundo is None:
        return None

    if not _normalizar_subpastas_projeto(pasta_projeto):
        logger.warning("[organização] A estrutura interna de %s teve conflitos; não vou mover a pasta ainda.", pasta_projeto)
        return None

    destino = fundo / pasta_projeto.name
    if pasta_projeto.resolve() == destino.resolve():
        logger.info("[organização] Projeto já está no lugar correto: %s", destino)
        return destino

    if destino.exists():
        logger.error(
            "[organização] Destino já existe e não está vazio: %s — projeto permanece em %s para evitar mistura.",
            destino,
            pasta_projeto,
        )
        return None

    try:
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(pasta_projeto), str(destino))
        logger.info("[organização] Projeto movido: %s -> %s", pasta_projeto, destino)
        return destino
    except OSError as exc:
        logger.error("[organização] Falha ao mover projeto %s -> %s: %s", pasta_projeto, destino, exc)
        return None
