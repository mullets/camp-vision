"""Testes de scanner.lote._atribuir_codigos_por_projeto — em especial
o cálculo de ano_pasta único por projeto, para não fragmentar um
mesmo projeto em várias pastas de ano diferentes."""

from pathlib import Path

from ai.interpretador import MetadadosPrancha
from scanner.lote import _atribuir_codigos_por_projeto


class _AnaliseFalsa:
    """Stub mínimo com o atributo .metadados, o único que
    _atribuir_codigos_por_projeto usa."""
    def __init__(self, metadados: MetadadosPrancha):
        self.metadados = metadados


def _lote(especificacoes: list[tuple[str, str, str]]) -> dict:
    """especificacoes: lista de (nome_arquivo, projeto, ano)."""
    analises_por_arquivo = {}
    for nome, projeto, ano in especificacoes:
        metadados = MetadadosPrancha(projeto=projeto, ano=ano, arquiteto="Fulano de Tal")
        analises_por_arquivo[Path(nome)] = [_AnaliseFalsa(metadados)]
    return analises_por_arquivo


def test_mesmo_projeto_com_anos_distintos_ganha_um_unico_ano_de_pasta():
    lote = _lote([
        ("a.tif", "Projeto X", ""),
        ("b.tif", "Projeto X", "1959"),
        ("c.tif", "Projeto X", "1959"),
        ("d.tif", "Projeto X", "1966"),
        ("e.tif", "Projeto X", "1927"),
    ])
    _atribuir_codigos_por_projeto(lote)

    anos_pasta = {analises[0].metadados.ano_pasta for analises in lote.values()}
    assert anos_pasta == {"1959"}


def test_ano_individual_de_cada_prancha_nao_e_alterado():
    lote = _lote([
        ("a.tif", "Projeto X", "1927"),
        ("b.tif", "Projeto X", "1959"),
        ("c.tif", "Projeto X", "1959"),
    ])
    _atribuir_codigos_por_projeto(lote)

    anos_individuais = sorted(analises[0].metadados.ano for analises in lote.values())
    assert anos_individuais == ["1927", "1959", "1959"]


def test_grupo_sem_nenhum_ano_legivel_fica_com_ano_pasta_vazio():
    lote = _lote([
        ("a.tif", "Projeto Y", ""),
        ("b.tif", "Projeto Y", ""),
    ])
    _atribuir_codigos_por_projeto(lote)

    anos_pasta = {analises[0].metadados.ano_pasta for analises in lote.values()}
    assert anos_pasta == {""}


def test_projetos_diferentes_tem_ano_de_pasta_independente():
    lote = _lote([
        ("a.tif", "Projeto X", "1959"),
        ("b.tif", "Projeto X", "1959"),
        ("c.tif", "Projeto Z", "2001"),
        ("d.tif", "Projeto Z", "2001"),
        ("e.tif", "Projeto Z", "1980"),
    ])
    _atribuir_codigos_por_projeto(lote)

    ano_por_projeto = {
        analises[0].metadados.projeto: analises[0].metadados.ano_pasta
        for analises in lote.values()
    }
    assert ano_por_projeto == {"Projeto X": "1959", "Projeto Z": "2001"}
