"""Testes unitários do extrator de metadados por regras (fallback sem IA)."""

from ai.fallback_regras import extrair_por_regras


def test_extrai_ano():
    texto = "PROJETO RESIDENCIAL\nANO: 2018\nESCALA 1:100"
    metadados = extrair_por_regras(texto)
    assert metadados.ano == "2018"


def test_extrai_escala():
    texto = "PLANTA BAIXA\nESCALA 1:75"
    metadados = extrair_por_regras(texto)
    assert "1:75" in metadados.escala.replace(" ", "")


def test_fonte_marcada_como_regras():
    metadados = extrair_por_regras("qualquer texto")
    assert metadados.fonte == "regras"
