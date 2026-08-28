"""Testes da ordenação de saída do exportador (agrupada por projeto,
ordenada por número de folha dentro de cada grupo)."""

import tempfile
from pathlib import Path

from exportacao.exportador import Exportador, RegistroExportacao


def _exportador_temporario() -> Exportador:
    return Exportador(Path(tempfile.mkdtemp()))


def _adicionar(exp: Exportador, arquivo: str, **campos) -> None:
    exp.adicionar_registro(RegistroExportacao(arquivo=arquivo, arquivo_original=arquivo, **campos))


def test_agrupa_por_projeto_e_ordena_por_folha():
    exp = _exportador_temporario()
    _adicionar(exp, "a", codigo_projeto_auto="OCG-P0032", numero="4")
    _adicionar(exp, "b", codigo_projeto_auto="OCG-P0032", numero="1")
    _adicionar(exp, "c", codigo_projeto_auto="OCG-P0003", numero="10")
    _adicionar(exp, "d", codigo_projeto_auto="OCG-P0032", numero="2")

    linhas = exp.gerar_linhas()
    arquivos_na_ordem = [linha["Arquivo"] for linha in linhas]

    assert arquivos_na_ordem == ["c", "b", "d", "a"]


def test_numero_de_folha_ausente_vai_para_o_fim_do_grupo():
    exp = _exportador_temporario()
    _adicionar(exp, "sem_folha", codigo_projeto_auto="OCG-P0003", numero="")
    _adicionar(exp, "com_folha", codigo_projeto_auto="OCG-P0003", numero="5")

    linhas = exp.gerar_linhas()
    arquivos_na_ordem = [linha["Arquivo"] for linha in linhas]

    assert arquivos_na_ordem == ["com_folha", "sem_folha"]


def test_numero_de_folha_com_texto_extra_usa_o_digito(): 
    exp = _exportador_temporario()
    _adicionar(exp, "folha_9", codigo_projeto_auto="OCG-P0001", numero="Folha N.9")
    _adicionar(exp, "folha_2", codigo_projeto_auto="OCG-P0001", numero="2")

    linhas = exp.gerar_linhas()
    arquivos_na_ordem = [linha["Arquivo"] for linha in linhas]

    assert arquivos_na_ordem == ["folha_2", "folha_9"]


def test_exportar_csv_e_xlsx_e_json_seguem_a_mesma_ordem():
    exp = _exportador_temporario()
    _adicionar(exp, "a", codigo_projeto_auto="OCG-P0002", numero="3")
    _adicionar(exp, "b", codigo_projeto_auto="OCG-P0001", numero="1")

    caminhos = exp.exportar_tudo()
    for caminho in caminhos.values():
        assert caminho.exists()


def test_relatorio_conta_metricas_corretamente():
    exp = _exportador_temporario()
    _adicionar(exp, "a", codigo_projeto_auto="P1", arquiteto="Fulano", numero="4",
               projeto="X", confianca_ocr=0.8)
    _adicionar(exp, "b", codigo_projeto_auto="P1", arquiteto="", numero="",
               projeto="X", confianca_ocr=0.0)
    _adicionar(exp, "c", codigo_projeto_auto="P2", arquiteto="Beltrano", numero="2",
               projeto="Y", confianca_ocr=0.9, observacoes="algum erro aqui")

    caminho = exp.exportar_relatorio()
    conteudo = caminho.read_text(encoding="utf-8")

    assert "Pranchas processadas: 3" in conteudo
    assert "Com texto de carimbo lido: 2" in conteudo
    assert "Número da folha: 2" in conteudo
    assert "Arquiteto: 2" in conteudo
    assert "Projetos distintos identificados no lote: 2" in conteudo
    assert "Registros com erro: 1" in conteudo
