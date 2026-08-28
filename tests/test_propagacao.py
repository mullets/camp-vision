"""Testes da propagação de metadados de identificação do projeto
entre pranchas do mesmo lote."""

import tempfile
from pathlib import Path

from exportacao.exportador import Exportador, RegistroExportacao
from scanner.propagacao import propagar_metadados_projeto


def _exportador_temporario() -> Exportador:
    return Exportador(Path(tempfile.mkdtemp()))


def _adicionar(exp: Exportador, nome: str, **campos) -> None:
    exp.adicionar_registro(RegistroExportacao(arquivo=nome, arquivo_original=nome, **campos))


def _por_nome(exp: Exportador, nome: str) -> RegistroExportacao:
    return next(r for r in exp.registros if r.arquivo_original == nome)


def test_valor_dominante_do_lote_preenche_pranchas_vazias():
    exp = _exportador_temporario()
    _adicionar(exp, "a.tif")
    _adicionar(exp, "b.tif", arquiteto="SAMI BUSSAB", ano="1974")
    _adicionar(exp, "c.tif", arquiteto="SAMI BUSSAB", ano="1974")

    propagar_metadados_projeto(exp)

    assert _por_nome(exp, "a.tif").arquiteto == "SAMI BUSSAB"
    assert _por_nome(exp, "a.tif").ano == "1974"
    assert "inferido" in _por_nome(exp, "a.tif").observacoes


def test_nunca_sobrescreve_valor_ja_preenchido():
    exp = _exportador_temporario()
    _adicionar(exp, "a.tif", ano="1980")
    _adicionar(exp, "b.tif", ano="1990")
    _adicionar(exp, "c.tif", ano="1990")

    propagar_metadados_projeto(exp)

    assert _por_nome(exp, "a.tif").ano == "1980"


def test_campo_divergente_herda_da_prancha_vizinha():
    """Cenário real (T100): várias obras diferentes na mesma pasta,
    digitalizadas em sequência — cada prancha sem carimbo deve herdar
    o endereço da obra vizinha, não o de uma obra qualquer do lote."""
    exp = _exportador_temporario()
    _adicionar(exp, "DEST2694.tif")
    _adicionar(exp, "DEST2696.tif", endereco="RUA PARAMU VILA ALPINA")
    _adicionar(exp, "DEST2697.tif", endereco="TRAV. DIOGO CALADO")
    _adicionar(exp, "DEST2698.tif")
    _adicionar(exp, "DEST2704.tif", endereco="VILA DEODORO")
    _adicionar(exp, "DEST2705.tif")

    propagar_metadados_projeto(exp)

    assert _por_nome(exp, "DEST2694.tif").endereco == "RUA PARAMU VILA ALPINA"
    assert _por_nome(exp, "DEST2698.tif").endereco == "TRAV. DIOGO CALADO"
    assert _por_nome(exp, "DEST2705.tif").endereco == "VILA DEODORO"


def test_ordem_e_natural_e_nao_alfabetica():
    """DEST9 vem antes de DEST10 (ordem de digitalização real)."""
    exp = _exportador_temporario()
    _adicionar(exp, "DEST9.tif", endereco="OBRA A")
    _adicionar(exp, "DEST10.tif")
    _adicionar(exp, "DEST11.tif", endereco="OBRA B")
    _adicionar(exp, "DEST12.tif", endereco="OBRA B")
    _adicionar(exp, "DEST13.tif", endereco="OBRA B")

    propagar_metadados_projeto(exp)

    assert _por_nome(exp, "DEST10.tif").endereco in ("OBRA A", "OBRA B")


def test_nao_propaga_campos_que_variam_por_prancha():
    exp = _exportador_temporario()
    _adicionar(exp, "a.tif", tipo="Planta", escala="1:50", prancha="1")
    _adicionar(exp, "b.tif", tipo="Corte", escala="1:100", prancha="2", projeto="Casa X")

    propagar_metadados_projeto(exp)

    assert _por_nome(exp, "a.tif").tipo == "Planta"
    assert _por_nome(exp, "a.tif").escala == "1:50"
    assert _por_nome(exp, "a.tif").prancha == "1"
    assert _por_nome(exp, "a.tif").projeto == "Casa X"


def test_nada_a_propagar_quando_tudo_vazio():
    exp = _exportador_temporario()
    _adicionar(exp, "a.tif")
    _adicionar(exp, "b.tif")

    assert propagar_metadados_projeto(exp) == 0
    assert _por_nome(exp, "a.tif").projeto == ""
