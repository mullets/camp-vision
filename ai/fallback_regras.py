"""
ai/fallback_regras.py
======================
Extração de metadados por regras (regex e palavras-chave), usada
como fallback quando a interpretação por IA não está disponível —
seja porque não há chave configurada, seja por indisponibilidade
temporária (ex.: cota da API excedida).

A correspondência de palavras-chave é APROXIMADA (tolerante a erros
de OCR), não uma busca por substring exata: folhas degradadas ou mal
digitalizadas raramente produzem a grafia perfeita de qualquer
palavra ("CLIENTE" pode virar "CUENTE", "OBRA" pode virar "0BRA"
etc.) — exigir correspondência exata faria o extrator simplesmente
não encontrar nada nesses casos, que são exatamente os casos em que
mais se precisa dele (a IA já lida bem com texto ruidoso; é este
extrator local, mais simples, que precisa compensar tolerando erro).

A precisão é naturalmente menor que a da IA, mas garante que o
pipeline continue produzindo uma catalogação básica mesmo sem acesso
à API da OpenAI.
"""

from __future__ import annotations

import difflib
import logging
import re

from ai.interpretador import MetadadosPrancha
from utils.texto import normalizar_maiusculas

logger = logging.getLogger("campvision.ai.regras")

# Aceita "1974" e também "1.974" — pranchas antigas brasileiras
# frequentemente escrevem o ano com ponto de milhar (visto em carimbos
# reais: "JULHO · 1.974"). O grupo opcional do ponto é descartado na
# normalização (ver `_normalizar_ano`).
_PADRAO_ANO = re.compile(r"\b([12])\.?(9|0)\d{2}\b")


ANO_MINIMO_PLAUSIVEL = 1850


def ano_valido(valor: str) -> bool:
    """Diz se um valor é realmente um ano de acervo arquitetônico.

    Existe porque valores inventados chegavam ao campo — num lote real
    apareceu "200K" em 6 pranchas, vindo de erro de OCR, e isso não
    fica só no CSV: o ano nomeia a PASTA de arquivamento, então uma
    leitura ruim cria uma pasta "200K" no acervo. É mais seguro deixar
    o campo vazio (e a prancha em "Ano desconhecido") do que registrar
    um ano que não existe."""
    valor = (valor or "").strip()
    if not valor.isdigit() or len(valor) != 4:
        return False
    from datetime import date
    return ANO_MINIMO_PLAUSIVEL <= int(valor) <= date.today().year


def _normalizar_ano(texto_encontrado: str) -> str:
    """Remove o ponto de milhar de um ano ("1.974" -> "1974")."""
    return texto_encontrado.replace(".", "")


# Muitos carimbos escrevem a data como mês + ano de DOIS dígitos
# ("FEVEREIRO/75", "JAN-77", "MAIO/76"). Sem isto, o ano dessas
# pranchas ficava vazio — e acabava preenchido erroneamente pela
# propagação com o ano de outra prancha do lote.
_MESES_ABREVIADOS = (
    "JAN", "FEV", "MAR", "ABR", "MAI", "JUN",
    "JUL", "AGO", "SET", "OUT", "NOV", "DEZ",
)
_PADRAO_ANO_CURTO = re.compile(
    r"\b(?:" + "|".join(m + r"[A-ZÇ]*" for m in _MESES_ABREVIADOS) + r")\s*[/\-.]\s*(\d{2})\b",
    re.IGNORECASE,
)


# Datas totalmente numéricas ("22/08/73", "05-12-1974") também
# aparecem nos carimbos, e o ano é sempre o ÚLTIMO campo. Sem isto o
# ano dessas pranchas ficava vazio e acabava preenchido pela
# propagação com o ano de outra prancha do lote.
_PADRAO_DATA_NUMERICA = re.compile(r"\b([0-3]?\d)[/\-.]([01]?\d)[/\-.]((?:19|20)?\d{2})\b")


def _expandir_ano_curto(dois_digitos: str) -> str:
    """Converte um ano de dois dígitos em quatro. Acervos de
    arquitetura moderna paulista são do século XX, então 30-99 vira
    19xx e 00-29 vira 20xx."""
    numero = int(dois_digitos)
    return f"19{dois_digitos}" if numero >= 30 else f"20{dois_digitos}"


def _corrigir_digitos_confundidos(texto: str) -> str:
    """Corrige confusões clássicas do OCR entre letras e dígitos
    dentro de sequências que são claramente numéricas — por exemplo,
    "I.974" ou "l.974" no lugar de "1.974" (visto em carimbos reais,
    onde o "1" em letra técnica é lido como "I" maiúsculo).

    Aplicado só como uma passada auxiliar antes de procurar o ano;
    não altera o texto original guardado na catalogação."""
    return re.sub(r"\b[Il](\.?\d{3})\b", r"1\1", texto)
_PADRAO_ESCALA = re.compile(r"\b1\s*[:/]\s*\d{1,4}\b")
_PADRAO_NUMERO_PRANCHA = re.compile(r"\b(?:PR|PRANCHA|FL|FOLHA)[\s.:\-]*\d+[/\-]?\d*\b", re.IGNORECASE)

# A "FOLHA" é o número que identifica a prancha dentro do projeto — é
# o que distingue uma folha da outra quando todo o resto do carimbo é
# igual. Em muitos carimbos o rótulo "FOLHA" fica numa linha e o
# número aparece isolado (às vezes em letra grande e estilizada, o que
# dificulta o OCR). Por isso tentamos vários formatos, do mais
# explícito ao mais frouxo.
_PADROES_FOLHA = (
    # "FOLHA 3", "FOLHA: 3", "FL. 3", "FOLHA N 3"
    re.compile(r"\b(?:FOLHA|FL)\b[\s.:\-]*(?:N[º°.]?\s*)?(\d{1,3})\b", re.IGNORECASE),
    # "FOLHA 3/12" — guarda só o número da folha
    re.compile(r"\b(?:FOLHA|FL)\b[\s.:\-]*(\d{1,3})\s*/\s*\d{1,3}\b", re.IGNORECASE),
    # "3 DE 12"
    re.compile(r"\b(\d{1,3})\s+DE\s+\d{1,3}\b", re.IGNORECASE),
)

# "FOLHA ÚNICA" é comum em projetos de uma prancha só.
_PADRAO_FOLHA_UNICA = re.compile(r"\bFOLHA\s+[UÚ]NICA\b", re.IGNORECASE)


def _extrair_folha(texto_ocr: str) -> str:
    """Extrai o número da folha (prancha) do texto do carimbo."""
    texto = _corrigir_digitos_confundidos(texto_ocr)

    if _PADRAO_FOLHA_UNICA.search(normalizar_maiusculas(texto)):
        return "ÚNICA"

    for padrao in _PADROES_FOLHA:
        encontrado = padrao.search(texto)
        if encontrado:
            return encontrado.group(1).lstrip("0") or "0"
    return ""

_PALAVRAS_CHAVE_CLIENTE = ("CLIENTE", "PROPRIETARIO")
_PALAVRAS_CHAVE_ARQUITETO = ("ARQUITETO", "AUTOR", "RESPONSAVEL TECNICO", "RESP TECNICO", "RT")
_PALAVRAS_CHAVE_ENDERECO = ("ENDERECO", "LOCAL", "RUA", "AV", "AVENIDA", "ALAMEDA")

# Endereços do ESCRITÓRIO do arquiteto, que aparecem no carimbo junto
# ao nome dele e NÃO são o endereço da obra. Sem esta lista, o
# endereço do escritório era extraído como se fosse o local do
# projeto — e pior, a propagação o espalhava por todo o acervo,
# apagando a informação que realmente distingue uma obra da outra.
# É uma lista de trechos: basta o trecho aparecer no valor extraído.
ENDERECOS_DE_ESCRITORIO = (
    "CONSELHEIRO TORRES HOMEM",
    "CONS. TORRES HOMEM",
    "CONS TORRES HOMEM",
    "TORRES HOMEM",
)


def _e_endereco_de_escritorio(valor: str) -> bool:
    """Diz se o endereço extraído é o do escritório do arquiteto (que
    consta no carimbo) em vez do endereço da obra."""
    normalizado = normalizar_maiusculas(valor)
    return any(trecho in normalizado for trecho in ENDERECOS_DE_ESCRITORIO)
_PALAVRAS_CHAVE_CIDADE = ("CIDADE", "MUNICIPIO", "LOCAL")
_PALAVRAS_CHAVE_PROJETO = ("PROJETO", "OBRA")
_PALAVRAS_CHAVE_FASE = ("FASE", "ANTEPROJETO", "EXECUTIVO", "PRELIMINAR", "ESTUDO PRELIMINAR")

LIMIAR_SIMILARIDADE = 0.72  # 0-1 (difflib) — tolera erros de OCR sem aceitar qualquer coisa


def _palavra_corresponde_a_chave(palavra: str, chave: str) -> bool:
    """Compara uma palavra do OCR com uma palavra-chave, tolerando
    erros de reconhecimento — mas exigindo tamanhos parecidos, para
    não confundir uma palavra curta e comum com uma chave mais longa
    que a contém como substring por coincidência (ex.: "SALA" não
    deve ser confundida com "ESCALA")."""
    if len(chave) < 3:
        return palavra == chave
    maior, menor = max(len(palavra), len(chave)), min(len(palavra), len(chave))
    if menor / maior < 0.75:
        return False
    return difflib.SequenceMatcher(None, palavra, chave).ratio() >= LIMIAR_SIMILARIDADE


_MESES = (
    "JANEIRO", "FEVEREIRO", "MARCO", "ABRIL", "MAIO", "JUNHO", "JULHO",
    "AGOSTO", "SETEMBRO", "OUTUBRO", "NOVEMBRO", "DEZEMBRO",
)


def _cortar_no_inicio_da_data(valor: str) -> str:
    """Corta o valor quando encontra um mês, um ano ou o rótulo de
    outro campo do carimbo.

    Em muitos carimbos vários campos caem na mesma linha do OCR (ex.:
    "ARQUITETO SAMI BUSSAB ... ESCALA 1:20 DATA FEVEREIRO/75"), e sem
    esse corte tudo isso acabava colado no fim do nome do arquiteto."""
    palavras = valor.split()
    for indice, palavra in enumerate(palavras):
        palavra_normalizada = normalizar_maiusculas(palavra).strip(".,;:-/")
        raiz = palavra_normalizada.split("/")[0].split("-")[0]
        e_data = (
            palavra_normalizada in _MESES
            or raiz in _MESES
            or raiz in _MESES_ABREVIADOS
            or _PADRAO_ANO.fullmatch(palavra_normalizada)
        )
        if e_data or palavra_normalizada in _ROTULOS_DE_OUTROS_CAMPOS:
            return " ".join(palavras[:indice]).strip(" :-\t.")
    return valor


# Rótulos que marcam o início de OUTRO campo do carimbo — ao topar
# com um deles, o valor do campo atual acabou.
_ROTULOS_DE_OUTROS_CAMPOS = frozenset({
    "ESCALA", "ESC", "DATA", "MODIF", "OBS", "FOLHA", "PRANCHA",
    "CLIENTE", "LOCAL", "OBRA", "PROJETO", "EXECUCAO", "TEL", "TELEFONE",
})


def _linha_apos_palavra_chave(linhas: list[str], chaves: tuple[str, ...]) -> str:
    """Procura, em cada linha do texto de OCR, uma palavra que
    corresponda aproximadamente a alguma das chaves (ex.: "CLIENTE"),
    e retorna o restante da linha após essa palavra — o valor do
    campo, presumivelmente."""
    for linha in linhas:
        palavras_linha = linha.split()
        for indice, palavra in enumerate(palavras_linha):
            palavra_normalizada = normalizar_maiusculas(palavra)
            if len(palavra_normalizada) < 3:
                continue
            for chave in chaves:
                if _palavra_corresponde_a_chave(palavra_normalizada, chave):
                    resto = " ".join(palavras_linha[indice + 1:]).strip(" :-\t.")
                    resto = _cortar_no_inicio_da_data(resto)
                    if resto:
                        return resto
    return ""


def _extrair_endereco_da_obra(linhas: list[str]) -> str:
    """Extrai o endereço da OBRA, descartando o do escritório."""
    endereco = _linha_apos_palavra_chave(linhas, _PALAVRAS_CHAVE_ENDERECO)
    if endereco and _e_endereco_de_escritorio(endereco):
        logger.debug("Endereço '%s' ignorado: é o endereço do escritório, não da obra.", endereco)
        return ""
    return endereco


def extrair_por_regras(texto_ocr: str) -> MetadadosPrancha:
    """Extrai metadados básicos do texto de OCR usando padrões
    regulares (ano, escala, número de prancha) e palavras-chave
    comuns em carimbos de arquitetura, com correspondência aproximada
    tolerante a erros de OCR."""
    linhas = [l.strip() for l in texto_ocr.splitlines() if l.strip()]

    texto_corrigido = _corrigir_digitos_confundidos(texto_ocr)
    ano_match = _PADRAO_ANO.search(texto_corrigido)
    # O ano de quatro dígitos tem prioridade; só se não houver é que
    # recorremos ao formato curto colado ao mês ("FEVEREIRO/75").
    ano_curto_match = None if ano_match else _PADRAO_ANO_CURTO.search(texto_corrigido)
    data_numerica = None if (ano_match or ano_curto_match) else _PADRAO_DATA_NUMERICA.search(texto_corrigido)
    escala_match = _PADRAO_ESCALA.search(texto_ocr)
    numero_match = _PADRAO_NUMERO_PRANCHA.search(texto_ocr)

    metadados = MetadadosPrancha(
        projeto=_linha_apos_palavra_chave(linhas, _PALAVRAS_CHAVE_PROJETO),
        cliente=_linha_apos_palavra_chave(linhas, _PALAVRAS_CHAVE_CLIENTE),
        arquiteto=_linha_apos_palavra_chave(linhas, _PALAVRAS_CHAVE_ARQUITETO),
        endereco=_extrair_endereco_da_obra(linhas),
        cidade=_linha_apos_palavra_chave(linhas, _PALAVRAS_CHAVE_CIDADE),
        ano=(
            _normalizar_ano(ano_match.group(0)) if ano_match
            else _expandir_ano_curto(ano_curto_match.group(1)) if ano_curto_match
            else (
                data_numerica.group(3) if len(data_numerica.group(3)) == 4
                else _expandir_ano_curto(data_numerica.group(3))
            ) if data_numerica
            else ""
        ),
        prancha="",
        numero=_extrair_folha(texto_ocr) or (numero_match.group(0) if numero_match else ""),
        escala=escala_match.group(0).replace(" ", "") if escala_match else "",
        fase=_linha_apos_palavra_chave(linhas, _PALAVRAS_CHAVE_FASE),
        tipo="",
        observacoes="",
        confianca_ia=0.4,  # confiança baixa/fixa por ser heurístico
        fonte="regras",
    )
    return metadados
