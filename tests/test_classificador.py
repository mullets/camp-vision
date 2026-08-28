"""Testes unitários do módulo de classificação automática."""

from classificacao.classificador import classificar


def test_classifica_planta_baixa():
    texto = "PLANTA BAIXA - PAVIMENTO TÉRREO\nESCALA 1:50"
    resultado = classificar(texto)
    assert resultado.tipo == "Planta"
    assert resultado.confianca > 0


def test_classifica_corte():
    texto = "CORTE AA\nCORTE BB\nESCALA 1:100"
    resultado = classificar(texto)
    assert resultado.tipo == "Corte"


def test_sem_evidencia_retorna_nao_classificado():
    resultado = classificar("texto qualquer sem relação")
    assert resultado.tipo == "Não classificado"
    assert resultado.confianca == 0.0


def test_sugestao_ia_influencia_resultado():
    resultado = classificar("texto ambíguo", tipo_sugerido_ia="Fachada")
    assert resultado.tipo == "Fachada"


def test_classifica_cortes_no_plural():
    """Carimbos reais escrevem "CORTES e VISTA" (plural)."""
    resultado = classificar("CORTES e VISTA   ESCALA 1:250")
    assert resultado.tipo == "Corte"
