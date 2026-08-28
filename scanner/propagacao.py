"""
scanner/propagacao.py
=======================
Preenche campos de identificação do projeto (projeto, cliente,
arquiteto, endereço, cidade, ano) que ficaram vazios numa prancha,
usando o que foi lido nas OUTRAS pranchas do mesmo lote.

Duas estratégias, nessa ordem:

1. VALOR DOMINANTE DO LOTE — quando um valor representa a maioria
   das pranchas que leram aquele campo, ele vale para todo o lote
   (ex.: o arquiteto e o ano costumam ser os mesmos em toda a pasta).

2. VIZINHANÇA — quando os valores divergem demais (a pasta contém
   obras diferentes), cada prancha vazia herda o valor da prancha
   mais próxima na ORDEM ORIGINAL DE DIGITALIZAÇÃO que tenha aquele
   campo preenchido. Pranchas de um mesmo projeto são digitalizadas
   em sequência, então o arquivo vizinho é quase sempre da mesma
   obra — muito melhor que carimbar o valor de uma obra qualquer do
   lote, e muito melhor que deixar vazio.

IMPORTANTE: nunca sobrescreve um valor já preenchido — só completa o
que estava genuinamente vazio. Campos que variam por prancha (número,
folha, título, escala, tipo, fase) nunca são propagados.
"""

from __future__ import annotations

import difflib
import logging
import re
from collections import Counter
from typing import TYPE_CHECKING, Any, Optional

from utils.texto import normalizar_maiusculas

if TYPE_CHECKING:  # evita import circular e dependências pesadas em tempo de execução
    from exportacao.exportador import RegistroExportacao

logger = logging.getLogger("campvision.propagacao")

# Campos de IDENTIFICAÇÃO DO PROJETO — tendem a ser os mesmos entre
# pranchas de um mesmo projeto. NUNCA inclua aqui campos que variam
# por prancha (número, folha, prancha/título, escala, tipo, fase).
CAMPOS_PROPAGAVEIS = ("projeto", "cliente", "arquiteto", "endereco", "cidade", "ano")

# Um valor vale para o lote inteiro se representar mais da metade das
# pranchas que leram aquele campo. Abaixo disso, assume-se que a pasta
# contém obras diferentes e usa-se a vizinhança (observado na prática:
# lote de praças da SABESP com endereços distintos na mesma pasta).
PROPORCAO_MINIMA_CONCORDANCIA = 0.5

# Um único registro NÃO faz maioria. Sem isso, um campo lido em uma
# só prancha do lote (1 de 1 = 100%) seria carimbado em todas as
# outras — observado na prática: um lote com uma prancha de outro
# projeto (Edifício IAMSPE) espalhou "RUA PEDRO DE TOLEDO" por todas
# as pranchas da Casa de Praia. Com no mínimo duas ocorrências, um
# valor solitário cai na regra de vizinhança, que é bem mais segura.
OCORRENCIAS_MINIMAS_PARA_DOMINAR = 2

# Distância máxima (em posições na ordem de digitalização) para herdar
# o valor de um vizinho. Além disso, é provável que já seja outro
# projeto dentro da mesma pasta.
DISTANCIA_MAXIMA_VIZINHO = 4


def _chave_ordenacao_natural(registro: "RegistroExportacao") -> tuple:
    """Ordena pelo nome ORIGINAL do arquivo, tratando números como
    números ("DEST9" antes de "DEST10") — assim a ordem reflete a
    sequência real de digitalização, e não a ordem alfabética crua nem
    a ordem (paralela, imprevisível) em que o lote foi processado."""
    nome = registro.arquivo_original or registro.arquivo
    return tuple(
        int(parte) if parte.isdigit() else parte.lower()
        for parte in re.split(r"(\d+)", nome)
    )


def _valor_dominante(registros: list, campo: str) -> Optional[str]:
    """Retorna o valor que representa a maioria das pranchas que
    leram este campo, ou None se os valores divergirem demais.

    A comparação ignora maiúsculas/minúsculas e acentos: o banco de
    conhecimento corrige o nome de alguns registros ("SAMI BUSSAB" →
    "Sami Bussab") e sem essa normalização as duas grafias contariam
    como valores diferentes, podendo derrubar a maioria e impedir a
    propagação de um campo que na verdade é o mesmo em todo o lote."""
    grafias: dict[str, Counter] = {}
    for registro in registros:
        valor = getattr(registro, campo, "").strip()
        if not valor:
            continue
        chave = normalizar_maiusculas(valor)
        grafias.setdefault(chave, Counter())[valor] += 1

    if not grafias:
        return None

    total = sum(sum(c.values()) for c in grafias.values())
    chave_dominante, contador = max(grafias.items(), key=lambda item: sum(item[1].values()))
    ocorrencias = sum(contador.values())

    if ocorrencias >= OCORRENCIAS_MINIMAS_PARA_DOMINAR and ocorrencias / total > PROPORCAO_MINIMA_CONCORDANCIA:
        # Entre as grafias equivalentes, usa a mais frequente.
        return contador.most_common(1)[0][0]

    # Duas razões diferentes levam até aqui, e confundi-las torna o
    # log inútil para diagnosticar: ou o campo foi lido em pouquíssimas
    # pranchas (sem respaldo para valer no lote inteiro), ou foi lido
    # em várias mas com valores conflitantes (pasta com projetos
    # diferentes).
    if len(grafias) == 1:
        logger.info(
            "Campo '%s' foi lido em apenas %d prancha(s) do lote — pouco para valer como valor "
            "de todo o lote; será usado o valor da prancha vizinha mais próxima.",
            campo, ocorrencias,
        )
    else:
        logger.info(
            "Campo '%s' tem valores conflitantes entre as pranchas (%d valores distintos, o mais "
            "comum em %d de %d) — provavelmente há projetos diferentes nesta pasta; será usado o "
            "valor da prancha vizinha mais próxima.",
            campo, len(grafias), ocorrencias, total,
        )
    return None


def _valor_do_vizinho_mais_proximo(
    registros_ordenados: list, posicao: int, campo: str,
) -> Optional[str]:
    """Procura, a partir de `posicao`, o valor preenchido mais próximo
    para `campo`, alternando entre o vizinho anterior e o seguinte."""
    for distancia in range(1, DISTANCIA_MAXIMA_VIZINHO + 1):
        for vizinho in (posicao - distancia, posicao + distancia):
            if 0 <= vizinho < len(registros_ordenados):
                valor = getattr(registros_ordenados[vizinho], campo, "").strip()
                if valor:
                    return valor
    return None


def propagar_metadados_projeto(registros_ou_exportador: Any) -> int:
    """Preenche os campos de identificação do projeto que ficaram
    vazios, usando o valor dominante do lote ou, quando os valores
    divergem, o da prancha vizinha na ordem de digitalização.

    Retorna quantos registros tiveram pelo menos um campo preenchido
    por propagação (para fins de log)."""
    # Aceita tanto o Exportador quanto uma lista simples de registros:
    # assim o avaliador (bench/) pode reaproveitar esta lógica sem
    # arrastar junto o exportador e suas dependências.
    brutos = getattr(registros_ou_exportador, "registros", registros_ou_exportador)
    registros = sorted(brutos, key=_chave_ordenacao_natural)
    if not registros:
        return 0

    dominantes = {
        campo: valor
        for campo in CAMPOS_PROPAGAVEIS
        if (valor := _valor_dominante(registros, campo)) is not None
    }

    # Os valores herdados são calculados ANTES de qualquer escrita, a
    # partir do estado original — senão um valor recém-propagado
    # serviria de fonte para o vizinho seguinte, e um único acerto
    # (ou erro) se espalharia em cadeia por todo o lote.
    preenchimentos: list[tuple[Any, str, str, str]] = []
    for posicao, registro in enumerate(registros):
        for campo in CAMPOS_PROPAGAVEIS:
            if getattr(registro, campo, "").strip():
                continue
            if campo in dominantes:
                preenchimentos.append((registro, campo, dominantes[campo], "lote"))
            else:
                valor = _valor_do_vizinho_mais_proximo(registros, posicao, campo)
                if valor:
                    preenchimentos.append((registro, campo, valor, "vizinha"))

    campos_por_registro: dict[int, list[str]] = {}
    for registro, campo, valor, origem in preenchimentos:
        setattr(registro, campo, valor)
        campos_por_registro.setdefault(id(registro), []).append(f"{campo} ({origem})")

    for registro in registros:
        campos = campos_por_registro.get(id(registro))
        if not campos:
            continue
        nota = f"[inferido de outra(s) prancha(s): {', '.join(campos)}]"
        registro.observacoes = f"{registro.observacoes} {nota}".strip() if registro.observacoes else nota

    total = len(campos_por_registro)
    if total:
        logger.info("Metadados de projeto completados em %d prancha(s) a partir das demais do lote.", total)
    return total


# Campos cujas grafias divergentes precisam ser unificadas antes de
# virarem pasta ou nome de arquivo.
CAMPOS_A_UNIFICAR = ("projeto", "cliente", "arquiteto", "endereco", "cidade")

# Similaridade a partir da qual dois valores são considerados o mesmo.
# Alta de propósito: unir coisas diferentes é pior que deixar duas
# grafias do mesmo item.
LIMIAR_MESMA_ENTIDADE = 0.86

# Similaridade exigida PALAVRA A PALAVRA ao detectar truncamento. Mais
# baixa que a do texto inteiro porque o erro costuma estar no fim da
# palavra ("PIRONO" por "PIRONDI"); a exigência de que TODAS as
# palavras casem é o que impede uniões indevidas.
SIMILARIDADE_PALAVRA_TRUNCADA = 0.75


def _e_versao_truncada(curto: str, longo: str) -> bool:
    """Diz se `curto` é uma versão incompleta de `longo`.

    Nem toda divergência é troca de letra: o OCR também corta o valor
    ("OSWALDO" em vez de "Oswaldo Correa Goncalves") ou erra o fim de
    um sobrenome ("CIRO PIRONO" por "Ciro Felice Pirondi"). Nesses
    casos a similaridade do texto inteiro é baixa demais para agrupar,
    mas TODAS as palavras do valor curto reaparecem no longo — que é o
    que verificamos aqui, tolerando erro de OCR palavra a palavra."""
    palavras_curto = normalizar_maiusculas(curto).split()
    palavras_longo = normalizar_maiusculas(longo).split()
    if not palavras_curto or len(palavras_curto) >= len(palavras_longo):
        return False
    return all(
        any(difflib.SequenceMatcher(None, pc, pl).ratio() >= SIMILARIDADE_PALAVRA_TRUNCADA for pl in palavras_longo)
        for pc in palavras_curto
    )


def unificar_grafias(registros_ou_exportador) -> int:
    """Faz variações do MESMO valor convergirem para uma grafia só.

    O OCR devolve o mesmo projeto ora como "CIA. TELEFONICA- GUARUJA",
    ora como "CIA. TELEFONICA - GUARUVA"; o mesmo cliente como "Lino
    Morgan" e "Lingo Morganti". Isso não fica só no CSV: o projeto e o
    ano NOMEIAM AS PASTAS de arquivamento, então cada variação racha um
    mesmo projeto em pastas diferentes, que é justamente o que torna o
    acervo organizado inútil.

    Valores parecidos são agrupados por similaridade textual e o grupo
    inteiro adota a grafia mais frequente — que, sendo a que mais
    apareceu, é a leitura mais confiável. Empates são resolvidos pela
    grafia mais longa, que costuma ser a menos truncada.

    Retorna quantos valores foram alterados."""
    brutos = getattr(registros_ou_exportador, "registros", registros_ou_exportador)
    alterados = 0

    for campo in CAMPOS_A_UNIFICAR:
        contagem: Counter = Counter()
        for registro in brutos:
            valor = getattr(registro, campo, "").strip()
            if valor:
                contagem[valor] += 1
        if len(contagem) < 2:
            continue

        # Grafias mais frequentes viram "âncoras"; as demais são
        # atraídas para a âncora parecida mais forte.
        ancoras: list[str] = []
        canonico: dict[str, str] = {}
        for valor, _ in sorted(contagem.items(), key=lambda kv: (-kv[1], -len(kv[0]))):
            alvo = next(
                (a for a in ancoras
                 if difflib.SequenceMatcher(
                     None, normalizar_maiusculas(valor), normalizar_maiusculas(a),
                 ).ratio() >= LIMIAR_MESMA_ENTIDADE
                 or _e_versao_truncada(valor, a)),
                None,
            )
            if alvo is None:
                ancoras.append(valor)
                canonico[valor] = valor
            else:
                canonico[valor] = alvo

        for registro in brutos:
            valor = getattr(registro, campo, "").strip()
            destino = canonico.get(valor)
            if destino and destino != valor:
                setattr(registro, campo, destino)
                alterados += 1
                logger.info("Grafia unificada em '%s': %r -> %r", campo, valor, destino)

    return alterados
