"""
utils/arquivamento.py
=======================
Organização física dos arquivos TIFF originais em pastas, seguindo a
lógica de classificação por ano e por projeto — prática alinhada aos
princípios de proveniência e organicidade tratados nas "Diretrizes
para o tratamento técnico de arquivos relacionados à arquitetura e ao
ambiente construído" (Resolução CONARQ/MGI nº 56, de 15/10/2024).

A resolução estabelece princípios arquivísticos gerais (manutenção da
unidade do fundo, respeito à proveniência, identificação por tipo
documental), não uma árvore de pastas literal — a estrutura abaixo é
a aplicação prática mais comum desses princípios em acervos de
arquitetura: separar por ano do projeto/prancha e, dentro dele, por
projeto (mantendo a unidade de cada fundo/projeto), com as pranchas
já renomeadas com código único dentro da pasta final.

Estrutura gerada (padrão):

    <raiz>/<ano>/<código do projeto> - <nome do projeto>/OCG-P0003-N0012 - ....tif

Onde `<código do projeto>` e `<nome do projeto>` vêm do projeto
identificado NAQUELA prancha (após propagação/unificação de grafias
entre as pranchas do lote — ver `scanner/propagacao.py`), não do nome
da pasta selecionada: um lote com vários projetos diferentes gera uma
pasta por projeto. Quando nenhum projeto é identificado para uma
prancha, usa-se o nome da pasta originalmente selecionada como
retaguarda (ver `scanner/pipeline._pasta_destino_final`).
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger("campvision.arquivamento")

_CARACTERES_PROIBIDOS_PASTA = re.compile(r"[/\\:*?\"<>|\x00]")
ANO_DESCONHECIDO = "Ano desconhecido"


def sanitizar_nome_pasta(texto: str, padrao: str = "Sem nome") -> str:
    """Remove caracteres inválidos para nomes de pasta no macOS,
    preservando espaços e acentos (diferente da sanitização de nomes
    de arquivo, mais permissiva aqui pois é só um nível de pasta)."""
    if not texto or not texto.strip():
        return padrao
    texto_limpo = _CARACTERES_PROIBIDOS_PASTA.sub("", texto.strip())
    return texto_limpo or padrao


def montar_pasta_destino(
    raiz: Path,
    padrao: str,
    ano: str,
    nome_projeto: str,
    codigo_projeto_auto: str = "",
) -> Path:
    """Monta o caminho da pasta final a partir do padrão configurado,
    sanitizando cada segmento.

    Por padrão, a pasta é organizada por CÓDIGO DE PROJETO
    (`{ano}/{codigo_projeto_auto} - {projeto}`) em vez de uma única
    pasta com o nome da pasta selecionada — assim, um lote com vários
    projetos diferentes (comum em acervos de escritório, digitalizados
    aos poucos) gera uma pasta por projeto, não uma pasta só com tudo
    misturado. `codigo_projeto_auto` é o código automático atribuído
    em `scanner/lote._atribuir_codigos_por_projeto` (ex.: "OCG-P0001");
    quando vazio (nenhum projeto foi identificado nesta prancha),
    `nome_projeto` cai para o nome da pasta selecionada (ver
    `scanner/pipeline._pasta_destino_final`), e o segmento de código
    fica de fora do nome da pasta em vez de aparecer vazio."""
    componentes = {
        "ano": sanitizar_nome_pasta(ano, ANO_DESCONHECIDO),
        "projeto": sanitizar_nome_pasta(nome_projeto, "Projeto sem nome"),
        "codigo_projeto_auto": sanitizar_nome_pasta(codigo_projeto_auto, ""),
    }
    try:
        segmentos = padrao.format(**componentes).split("/")
    except (KeyError, IndexError) as exc:
        logger.error("Padrão de pastas inválido '%s' (%s). Usando padrão de segurança.", padrao, exc)
        segmentos = [componentes["ano"], componentes["projeto"]]

    caminho = raiz
    for segmento in segmentos:
        segmento = _limpar_segmento_pasta(segmento.strip())
        if segmento:
            caminho = caminho / segmento
    return caminho


def _limpar_segmento_pasta(segmento: str) -> str:
    """Remove separadores " - " que sobrariam vazios quando um dos
    componentes do nome da pasta (tipicamente {codigo_projeto_auto},
    quando nenhum projeto foi identificado) não foi preenchido —
    evita pastas como "2018/ - Projeto X" em vez de "2018/Projeto X"."""
    partes = [p.strip() for p in segmento.split(" - ")]
    return " - ".join(p for p in partes if p).strip("-_ ")


def arquivar_em_pasta_destino(
    caminho_arquivo: Path, pasta_destino: Path, copiar: bool = True, novo_nome: Optional[str] = None,
) -> Path:
    """Leva o arquivo para a pasta de arquivamento final, criando a
    estrutura de pastas se necessário — em uma ÚNICA operação de
    disco (copiar ou mover diretamente do local original para o
    destino final, já com o nome final se `novo_nome` for informado).

    Isso é importante: fazer "renomear, depois mover" em duas etapas
    separadas significa que a primeira etapa MUTA o arquivo original
    no lugar — e se o mesmo arquivo for reprocessado depois (ex.: com
    um código de projeto diferente), o nome já modificado vira a base
    do próximo nome, crescendo a cada execução. Fazendo tudo numa
    operação só, a partir do caminho original de verdade, esse
    problema não pode acontecer.

    Por padrão COPIA (`copiar=True`) em vez de mover — o arquivo
    original continua intacto em seu local e nome originais, e uma
    cópia (já renomeada, se aplicável) vai para a pasta ano/projeto.
    Passe `copiar=False` para mover de verdade, se preferir economizar
    espaço em disco e já confiar no fluxo.

    Retorna o novo caminho (o da cópia/arquivo movido), ou o caminho
    original se nada precisou ser feito ou se algo falhou (registra o
    erro nesse caso)."""
    nome_final = novo_nome or caminho_arquivo.name
    if caminho_arquivo.parent == pasta_destino and nome_final == caminho_arquivo.name:
        return caminho_arquivo

    try:
        pasta_destino.mkdir(parents=True, exist_ok=True)
        destino_final = pasta_destino / nome_final

        if destino_final.exists():
            logger.warning(
                "Já existe um arquivo em '%s' — mantendo '%s' em seu local original para evitar sobrescrita.",
                destino_final, caminho_arquivo.name,
            )
            return caminho_arquivo

        if copiar:
            shutil.copy2(str(caminho_arquivo), str(destino_final))
            logger.info("Arquivo copiado para pasta de arquivamento: %s (original mantido em %s)",
                        destino_final, caminho_arquivo)
        else:
            shutil.move(str(caminho_arquivo), str(destino_final))
            logger.info("Arquivo movido para pasta de arquivamento: %s", destino_final)
        return destino_final
    except OSError as exc:
        logger.error(
            "Falha ao levar '%s' para a pasta de arquivamento '%s' (%s). Mantendo local original.",
            caminho_arquivo.name, pasta_destino, exc,
        )
        return caminho_arquivo


# Nome anterior mantido como alias, por compatibilidade.
mover_para_pasta_de_arquivamento = arquivar_em_pasta_destino
