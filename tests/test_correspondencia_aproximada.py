"""Testes unitários da correspondência aproximada de palavras-chave
(tolerante a erros de OCR), usada tanto na verificação de conteúdo do
carimbo quanto no extrator de metadados por regras."""

from ai.fallback_regras import extrair_por_regras
from scanner.detector_carimbo import _contar_acertos_aproximados, _pontuar_conteudo
from utils.texto import normalizar_maiusculas


def test_reconhece_palavra_chave_com_erro_de_ocr():
    texto = normalizar_maiusculas("TELEF0NE 853 4530, ARQUITET0 responsavel")
    assert _contar_acertos_aproximados(texto) >= 2


def test_nao_reconhece_texto_sem_relacao():
    texto = normalizar_maiusculas("porta janela sala quarto banheiro")
    assert _contar_acertos_aproximados(texto) == 0


def test_pontuacao_conteudo_maior_com_texto_de_carimbo_degradado():
    texto_carimbo_degradado = "TELEF0NE 853-4530 ARQUITET0 CUENTE"
    texto_planta = "DORMITORIO SALA COZINHA BANHEIRO CIRCULACAO"
    assert _pontuar_conteudo(texto_carimbo_degradado) > _pontuar_conteudo(texto_planta)


def test_extrator_por_regras_tolera_erro_de_ocr():
    texto = "0BRA MORUMBI BUSINESS\nCUENTE OGGI CONSTR EMPR IMOB\n1:50\n1988"
    metadados = extrair_por_regras(texto)
    assert metadados.projeto == "MORUMBI BUSINESS"
    assert "OGGI" in metadados.cliente
    assert metadados.escala == "1:50"
    assert metadados.ano == "1988"


def test_extrai_ano_com_ponto_de_milhar():
    """Pranchas antigas brasileiras escrevem o ano como "1.974"."""
    metadados = extrair_por_regras("ARQUITETOS SAMI BUSSAB   JULHO 1.974")
    assert metadados.ano == "1974"


def test_extrai_ano_mesmo_com_ocr_lendo_1_como_I():
    metadados = extrair_por_regras("ARQUITETOS SAMI BUSSAB   JULHO I.974")
    assert metadados.ano == "1974"


def test_nao_confunde_escala_com_ano():
    metadados = extrair_por_regras("PLANTA ESCALA 1:250")
    assert metadados.ano == ""
    assert metadados.escala == "1:250"


def test_data_na_mesma_linha_nao_gruda_no_arquiteto():
    metadados = extrair_por_regras("ARQUITETOS SAMI BUSSAB SATORU NAGAI   JULHO 1.974")
    assert metadados.arquiteto == "SAMI BUSSAB SATORU NAGAI"
