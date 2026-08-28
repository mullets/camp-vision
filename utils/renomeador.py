"""
utils/renomeador.py
=====================
Geração do nome de arquivo final de cada prancha, seguindo um padrão
de arquivamento comum em escritórios de arquitetura: um código curto
do projeto, o identificador da prancha (extraído do carimbo) e um
número sequencial único de controle, por exemplo:

    SB-P001.1-00001.tif
    │  │       └── sequencial único (evita colisão mesmo se duas
    │  │           pranchas tiverem o mesmo identificador, ex.:
    │  │           revisões ou numeração duplicada no acervo original)
    │  └── identificador da prancha (do carimbo, ex. "P001.1")
    └── código do projeto (ex. "SB" para "Sami Bussab")

O padrão exato é configurável (`config.settings.renomeacao_padrao`),
usando os placeholders `{codigo_projeto}`, `{prancha}`, `{sequencial}`,
`{tipo}` e `{ano}`, entre outros. O número de dígitos do sequencial
também é configurável (`renomeacao_digitos_sequencial`, padrão 5 →
"00001").

O padrão DEFAULT, no entanto, não usa código manual nem sequencial
simples — usa o código automático por projeto (atribuído em
`scanner/lote._atribuir_codigos_por_projeto`, já com prefixo tirado
do arquiteto/cidade do carimbo) e o título da prancha:

    {codigo_projeto_auto}-{sequencial_no_projeto} - {projeto} - {nome_prancha}

    OCG-P0003-N0012 - TEATRO DE SANTOS - PLANTA 01
    │   │      │       │                  └── título da prancha, do
    │   │      │       │                      carimbo, ou tipo+número
    │   │      │       │                      quando o carimbo não
    │   │      │       │                      trouxe título legível
    │   │      │       └── nome do projeto (propagado/unificado entre
    │   │      │           as pranchas do lote)
    │   │      └── posição da prancha dentro do projeto
    │   └── projeto nº3 do lote (ordem de digitalização)
    └── prefixo automático (iniciais do arquiteto/escritório do
        carimbo, ex. "OCG" para "Oswaldo Correa Goncalves")
"""

from __future__ import annotations

import logging
import re
import threading

logger = logging.getLogger("campvision.renomeador")

_PADRAO_CARACTERES_INVALIDOS = re.compile(r"[^A-Za-z0-9._-]+")
_CARACTERES_PROIBIDOS_ARQUIVO = re.compile(r'[/\\:*?"<>|\x00]')


def sanitizar_componente(texto: str, padrao: str = "SEMCODIGO") -> str:
    """Remove espaços e caracteres inválidos em nomes de arquivo,
    mantendo apenas letras, números, ponto, traço e underscore —
    suficiente para preservar identificadores como 'P001.1'."""
    if not texto or not texto.strip():
        return padrao
    texto_limpo = texto.strip().replace(" ", "")
    texto_limpo = _PADRAO_CARACTERES_INVALIDOS.sub("", texto_limpo)
    return texto_limpo or padrao


def sanitizar_texto_legivel(texto: str, padrao: str = "") -> str:
    """Sanitização mais PERMISSIVA que `sanitizar_componente`: remove
    só os caracteres proibidos em nomes de arquivo no macOS, mas
    PRESERVA espaços — usada nos componentes de texto legível do nome
    (nome do projeto, nome da prancha), que aparecem separados por
    " - " no padrão default (ex.: "TEATRO DE SANTOS", "PLANTA 01"),
    diferente dos componentes de CÓDIGO (codigo_projeto, sequencial),
    que continuam compactos, sem espaço."""
    if not texto or not texto.strip():
        return padrao
    texto_limpo = _CARACTERES_PROIBIDOS_ARQUIVO.sub("", texto.strip())
    texto_limpo = re.sub(r"\s+", " ", texto_limpo).strip()
    return texto_limpo or padrao


def sugerir_codigo_projeto(nome_pasta: str) -> str:
    """Sugere um código curto de projeto a partir do nome da pasta
    selecionada, usando as iniciais das palavras (ex.: 'Sami Bussab'
    -> 'SB'). Serve apenas como valor inicial editável pelo usuário."""
    palavras = [p for p in re.split(r"[\s_\-]+", nome_pasta.strip()) if p]
    if not palavras:
        return "PROJ"
    iniciais = "".join(p[0].upper() for p in palavras if p[0].isalnum())
    return iniciais[:6] or "PROJ"


def sugerir_prefixo_projeto(arquiteto: str = "", cidade: str = "") -> str:
    """Sugere um prefixo curto para o CÓDIGO AUTOMÁTICO de projeto
    (ex.: {codigo_projeto_auto} = "OCG-P0001"), extraído das iniciais
    do ARQUITETO/escritório lido do carimbo — ou, na ausência dele, da
    CIDADE — para que o código já comunique de qual acervo se trata
    sem depender do nome da pasta selecionada nem de configuração
    manual.

    Ao contrário de `sugerir_codigo_projeto` (usado como valor inicial
    EDITÁVEL na tela de Configurações, para o código MANUAL do
    projeto), esta função nunca inventa um valor de preenchimento: se
    não houver arquiteto nem cidade lidos, retorna string vazia, e o
    código automático do projeto cai para "P0001" sem prefixo."""
    fonte = (arquiteto or "").strip() or (cidade or "").strip()
    if not fonte:
        return ""
    palavras = [p for p in re.split(r"[\s_\-]+", fonte) if p]
    iniciais = "".join(p[0].upper() for p in palavras if p[0].isalnum())
    return iniciais[:6]


def montar_nome_prancha(titulo: str, tipo: str, numero: str) -> str:
    """Monta o nome legível da prancha para o nome do arquivo (ex.:
    "PLANTA HIDRÁULICA 03", "CORTE CC").

    Prioriza o TÍTULO lido do próprio carimbo (`prancha`), que já vem
    com o identificador da prancha do jeito que o escritório escreveu.
    Só cai para TIPO + NÚMERO (ex.: "Planta 01") quando o carimbo não
    trouxe um título legível — nesse caso o tipo detectado pela
    classificação automática serve de identificação mínima do
    conteúdo, em vez de deixar o nome do arquivo sem nenhuma pista."""
    titulo = (titulo or "").strip()
    if titulo:
        return titulo
    tipo = (tipo or "").strip()
    if not tipo or tipo.lower() == "não classificado":
        return ""
    numero = (numero or "").strip()
    return f"{tipo} {numero}".strip()


def montar_nome_arquivo(
    padrao: str,
    codigo_projeto: str,
    prancha: str,
    sequencial: int,
    extensao: str,
    digitos_sequencial: int = 5,
    tipo: str = "",
    ano: str = "",
    arquiteto: str = "",
    endereco: str = "",
    cliente: str = "",
    fase: str = "",
    projeto: str = "",
    folha: str = "",
    codigo_projeto_auto: str = "",
    sequencial_no_projeto: int = 0,
    nome_prancha: str = "",
) -> str:
    """Monta o nome de arquivo final a partir do padrão configurado e
    dos componentes extraídos da prancha, sanitizando cada componente
    para garantir um nome de arquivo válido em macOS.

    Campos disponíveis no padrão: {codigo_projeto}, {prancha},
    {folha}, {sequencial}, {tipo}, {ano}, {arquiteto}, {endereco},
    {cliente}, {fase}, {projeto}, {codigo_projeto_auto},
    {sequencial_no_projeto}, {nome_prancha}.

    {codigo_projeto_auto} é o código atribuído automaticamente a cada
    projeto distinto encontrado no lote, já com o prefixo tirado do
    arquiteto/cidade do carimbo quando disponível (ex.: "OCG-P0001"),
    e {sequencial_no_projeto} é a posição da prancha DENTRO do seu
    projeto (ex.: "N0012") — juntos formam nomes como
    "OCG-P0001-N0012", que identificam projeto e folha sem depender do
    nome original do arquivo. {nome_prancha} é o título/identificador
    da prancha (ex.: "PLANTA 01", "CORTE CC") — ver
    `montar_nome_prancha`.

    Campos de texto livre CURTOS (arquiteto, endereço, cliente) têm
    espaços removidos e são encurtados, para caber compactos no nome.
    {projeto} e {nome_prancha} usam sanitização mais permissiva
    (preservam espaços — ver `sanitizar_texto_legivel`), pensada para
    aparecer por extenso no padrão default, separados por " - "; o
    dado completo, de qualquer forma, sempre fica no CSV e nos
    metadados EXIF."""
    componentes = {
        "codigo_projeto": sanitizar_componente(codigo_projeto, "PROJ"),
        "prancha": sanitizar_componente(prancha, "SEMNUM"),
        "folha": sanitizar_componente(folha, "") or "",
        "sequencial": str(sequencial).zfill(digitos_sequencial),
        "tipo": sanitizar_componente(tipo, "") or "",
        "ano": sanitizar_componente(ano, "") or "",
        "arquiteto": sanitizar_componente(_encurtar(arquiteto), "") or "",
        "endereco": sanitizar_componente(_encurtar(endereco), "") or "",
        "cliente": sanitizar_componente(_encurtar(cliente), "") or "",
        "projeto": sanitizar_texto_legivel(_encurtar(projeto), "") or "",
        "fase": sanitizar_componente(fase, "") or "",
        # Sem código automático (ex.: processamento avulso, fora do
        # fluxo de lote que atribui P0001/P0002...), cai para o
        # código manual configurado, e só então para "PROJ" — nunca
        # fica vazio, o que deixaria um traço solto no nome.
        "codigo_projeto_auto": sanitizar_componente(codigo_projeto_auto or codigo_projeto, "PROJ"),
        "sequencial_no_projeto": f"N{sequencial_no_projeto:04d}" if sequencial_no_projeto else "",
        "nome_prancha": sanitizar_texto_legivel(_encurtar(nome_prancha), "") or "",
    }
    try:
        nome_sem_extensao = padrao.format(**componentes)
    except (KeyError, IndexError) as exc:
        logger.error("Padrão de nomenclatura inválido '%s' (%s). Usando padrão de segurança.", padrao, exc)
        nome_sem_extensao = f"{componentes['codigo_projeto']}-{componentes['prancha']}-{componentes['sequencial']}"

    nome_sem_extensao = _limpar_separadores(nome_sem_extensao)

    extensao_normalizada = extensao if extensao.startswith(".") else f".{extensao}"
    return f"{nome_sem_extensao}{extensao_normalizada.lower()}"


def _limpar_separadores(nome: str) -> str:
    """Remove separadores que sobram vazios quando um dos componentes
    do padrão não foi preenchido — sem isto um padrão como
    "{codigo_projeto_auto}-{sequencial_no_projeto} - {projeto} -
    {nome_prancha}" com {projeto} vazio geraria algo como
    "OCG-P0001-N0012 -  - PLANTA 01" em vez de
    "OCG-P0001-N0012 - PLANTA 01".

    Trata os dois estilos de separador usados no projeto: o traço
    compacto ("-", sem espaço, entre componentes de código) e o traço
    "legível" (" - ", com espaço, entre componentes de texto)."""
    # Segmentos separados por " - ": cada um é limpo individualmente
    # (colapsando/removendo traços soltos nas pontas) e os que ficam
    # vazios são descartados antes de juntar de novo.
    segmentos = [
        re.sub(r"[-_]{2,}", "-", segmento).strip("-_ ")
        for segmento in nome.split(" - ")
    ]
    nome_limpo = " - ".join(segmento for segmento in segmentos if segmento)
    return nome_limpo.strip("-_ ")


LIMITE_COMPONENTE_TEXTO = 30


def _encurtar(valor: str, limite: int = LIMITE_COMPONENTE_TEXTO) -> str:
    """Encurta um componente de texto livre para manter o nome de
    arquivo utilizável. Corta em limite de palavra quando possível."""
    if len(valor) <= limite:
        return valor
    cortado = valor[:limite]
    if " " in cortado:
        cortado = cortado[:cortado.rfind(" ")]
    return cortado.strip()


def parece_ja_renomeado(nome_sem_extensao: str, codigo_projeto: str, digitos_sequencial: int) -> bool:
    """Detecta se um nome de arquivo já parece ter sido gerado por uma
    execução anterior do CAMP Vision para o mesmo código de projeto —
    ou seja, começa com '{codigo_projeto}-' e termina em '-NNNNN' (o
    sequencial, com a quantidade de dígitos configurada).

    Usada tanto na varredura de arquivos (para não reprocessar
    arquivos já processados, ex. se eles ainda estiverem dentro da
    pasta selecionada) quanto na renomeação (proteção redundante) —
    sem essa checagem, reprocessar acidentalmente um arquivo já
    renomeado faria o nome crescer a cada execução."""
    if not codigo_projeto:
        return False
    padrao = re.compile(rf"^{re.escape(codigo_projeto)}-.+-\d{{{digitos_sequencial}}}$")
    return bool(padrao.match(nome_sem_extensao))


class GeradorSequencial:
    """Fornece números sequenciais únicos por código de projeto,
    thread-safe, para uso durante um processamento em lote com várias
    threads concorrentes.

    Mantém sua própria sessão de banco (protegida por um lock interno,
    já que uma sessão SQLAlchemy não é thread-safe por si só) e
    persiste cada número reservado imediatamente, de forma que o
    contador nunca reinicie nem colida entre execuções diferentes do
    programa — inclusive se o processamento for cancelado no meio.
    """

    def __init__(self, caminho_db: str):
        from database.models import criar_sessao  # import tardio: evita ciclo de import

        self._sessao = criar_sessao(caminho_db)
        self._lock = threading.Lock()

    def proximo(self, codigo_projeto: str) -> int:
        from database.repository import SequenciaRepository

        with self._lock:
            repositorio = SequenciaRepository(self._sessao)
            numero = repositorio.proximo_numero(codigo_projeto)
        logger.debug("Sequencial reservado para '%s': %d", codigo_projeto, numero)
        return numero

    def fechar(self) -> None:
        self._sessao.close()
