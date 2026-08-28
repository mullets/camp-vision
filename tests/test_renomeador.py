"""Testes unitários da geração de nomes de arquivo (renomeação com código único)."""

from utils.renomeador import (
    montar_nome_arquivo, montar_nome_prancha, sanitizar_componente,
    sugerir_codigo_projeto, sugerir_prefixo_projeto,
)


def test_exemplo_do_usuario_sami_bussab():
    nome = montar_nome_arquivo(
        padrao="{codigo_projeto}-{prancha}-{sequencial}",
        codigo_projeto="SB",
        prancha="P001.1",
        sequencial=1,
        extensao=".tif",
        digitos_sequencial=5,
    )
    assert nome == "SB-P001.1-00001.tif"


def test_sequencial_com_padding_customizado():
    nome = montar_nome_arquivo(
        padrao="{codigo_projeto}-{prancha}-{sequencial}",
        codigo_projeto="SB",
        prancha="P002",
        sequencial=42,
        extensao=".tiff",
        digitos_sequencial=5,
    )
    assert nome == "SB-P002-00042.tiff"


def test_sugere_codigo_a_partir_do_nome_da_pasta():
    assert sugerir_codigo_projeto("Sami Bussab") == "SB"
    assert sugerir_codigo_projeto("Residencial Vila Verde") == "RVV"


def test_sanitiza_caracteres_invalidos():
    resultado = sanitizar_componente("Prancha nº 01/A")
    assert "/" not in resultado
    assert " " not in resultado


def test_componente_vazio_usa_valor_padrao():
    assert sanitizar_componente("", "SEMNUM") == "SEMNUM"
    assert sanitizar_componente("   ", "SEMNUM") == "SEMNUM"


def test_padrao_invalido_cai_para_seguranca():
    nome = montar_nome_arquivo(
        padrao="{campo_que_nao_existe}",
        codigo_projeto="SB",
        prancha="P001",
        sequencial=1,
        extensao=".tif",
    )
    assert nome.startswith("SB-P001-")


def test_novos_campos_no_padrao_de_nome():
    nome = montar_nome_arquivo(
        padrao="{codigo_projeto}-{ano}-{tipo}-{folha}-{sequencial}",
        codigo_projeto="SABESP", prancha="", sequencial=9, extensao=".tif",
        ano="1974", tipo="Planta", folha="1",
    )
    assert nome == "SABESP-1974-Planta-1-00009.tif"


def test_campo_vazio_nao_deixa_separador_duplicado():
    nome = montar_nome_arquivo(
        padrao="{codigo_projeto}-{folha}-{sequencial}",
        codigo_projeto="SABESP", prancha="", sequencial=9, extensao=".tif", folha="",
    )
    assert nome == "SABESP-00009.tif"


def test_texto_livre_e_encurtado_no_nome():
    nome = montar_nome_arquivo(
        padrao="{codigo_projeto}-{endereco}-{sequencial}",
        codigo_projeto="SABESP", prancha="", sequencial=9, extensao=".tif",
        endereco="RUA CEL. RODRIGUES, VILA FORMOSA - S. PAULO",
    )
    assert len(nome) < 60
    assert nome.startswith("SABESP-RUACEL")


def test_prefixo_automatico_prioriza_arquiteto():
    assert sugerir_prefixo_projeto("Oswaldo Correa Goncalves", "Santos") == "OCG"


def test_prefixo_automatico_cai_para_cidade_sem_arquiteto():
    assert sugerir_prefixo_projeto("", "Santos") == "S"
    assert sugerir_prefixo_projeto("", "São Paulo") == "SP"


def test_prefixo_automatico_vazio_sem_arquiteto_nem_cidade():
    assert sugerir_prefixo_projeto("", "") == ""


def test_nome_prancha_prioriza_titulo_do_carimbo():
    assert montar_nome_prancha("PLANTA HIDRAULICA 03", "Hidráulica", "3") == "PLANTA HIDRAULICA 03"


def test_nome_prancha_cai_para_tipo_e_numero_sem_titulo():
    assert montar_nome_prancha("", "Planta", "01") == "Planta 01"


def test_nome_prancha_vazio_sem_titulo_nem_tipo_classificado():
    assert montar_nome_prancha("", "Não classificado", "01") == ""
    assert montar_nome_prancha("", "", "01") == ""


def test_padrao_default_com_codigo_automatico_projeto_e_prancha():
    """Reproduz o exemplo dado pelo usuário:
    "OCG-P0003-N0012 - TEATRO DE SANTOS - PLANTA 01"."""
    nome = montar_nome_arquivo(
        padrao="{codigo_projeto_auto}-{sequencial_no_projeto} - {projeto} - {nome_prancha}",
        codigo_projeto="", prancha="", sequencial=12, extensao=".tif",
        codigo_projeto_auto="OCG-P0003", sequencial_no_projeto=12,
        projeto="TEATRO DE SANTOS", nome_prancha="PLANTA 01",
    )
    assert nome == "OCG-P0003-N0012 - TEATRO DE SANTOS - PLANTA 01.tif"


def test_padrao_default_sem_projeto_nem_nome_prancha_nao_deixa_separador_solto():
    nome = montar_nome_arquivo(
        padrao="{codigo_projeto_auto}-{sequencial_no_projeto} - {projeto} - {nome_prancha}",
        codigo_projeto="", prancha="", sequencial=1, extensao=".tif",
        codigo_projeto_auto="P0001", sequencial_no_projeto=1,
    )
    assert nome == "P0001-N0001.tif"


def test_padrao_default_so_com_projeto_nao_deixa_separador_solto_no_fim():
    nome = montar_nome_arquivo(
        padrao="{codigo_projeto_auto}-{sequencial_no_projeto} - {projeto} - {nome_prancha}",
        codigo_projeto="", prancha="", sequencial=1, extensao=".tif",
        codigo_projeto_auto="P0001", sequencial_no_projeto=1, projeto="Casa X",
    )
    assert nome == "P0001-N0001 - Casa X.tif"


def test_codigo_projeto_auto_ausente_cai_para_codigo_manual():
    """Processamento avulso, fora do fluxo de lote que atribui
    P0001/P0002... — sem código automático, cai pro código manual
    configurado em vez de deixar o nome quebrado."""
    nome = montar_nome_arquivo(
        padrao="{codigo_projeto_auto}-{sequencial_no_projeto} - {projeto} - {nome_prancha}",
        codigo_projeto="SB", prancha="", sequencial=1, extensao=".tif",
        projeto="Casa X", nome_prancha="Planta 01",
    )
    assert nome == "SB - Casa X - Planta 01.tif"


def test_projeto_e_nome_prancha_preservam_espacos_no_nome():
    nome = montar_nome_arquivo(
        padrao="{codigo_projeto_auto} - {projeto} - {nome_prancha}",
        codigo_projeto="", prancha="", sequencial=1, extensao=".tif",
        codigo_projeto_auto="OCG-P0001", projeto="TEATRO DE SANTOS", nome_prancha="CORTE CC",
    )
    assert nome == "OCG-P0001 - TEATRO DE SANTOS - CORTE CC.tif"
