"""Testes unitários da organização em pastas (ano/projeto), alinhada às
diretrizes da Resolução CONARQ/MGI nº 56/2024 para acervos de arquitetura."""

from pathlib import Path

from utils.arquivamento import ANO_DESCONHECIDO, montar_pasta_destino, sanitizar_nome_pasta


def test_monta_pasta_ano_projeto():
    destino = montar_pasta_destino(Path("/acervo"), "{ano}/{projeto}", "2018", "Sami Bussab")
    assert destino == Path("/acervo/2018/Sami Bussab")


def test_ano_ausente_usa_pasta_padrao():
    destino = montar_pasta_destino(Path("/acervo"), "{ano}/{projeto}", "", "Sami Bussab")
    assert destino == Path(f"/acervo/{ANO_DESCONHECIDO}/Sami Bussab")


def test_sanitiza_nome_de_pasta_mantendo_espacos():
    resultado = sanitizar_nome_pasta("Sami Bussab: Projeto/Residência")
    assert "/" not in resultado
    assert ":" not in resultado
    assert " " in resultado  # espaços são permitidos em nomes de pasta


def test_padrao_customizado_com_ordem_invertida():
    destino = montar_pasta_destino(Path("/acervo"), "{projeto}/{ano}", "2020", "Casa X")
    assert destino == Path("/acervo/Casa X/2020")


def test_reprocessar_com_codigo_diferente_nao_muda_nome_original(tmp_path):
    """Reproduz o bug real observado: reprocessar o mesmo arquivo com
    um código de projeto diferente não pode fazer o nome do arquivo
    original crescer/compor a cada execução."""
    from utils.arquivamento import arquivar_em_pasta_destino

    original = tmp_path / "Copia de DEST2977.tif"
    original.write_bytes(b"conteudo do scan original")
    pasta_destino = tmp_path / "saida"

    r1 = arquivar_em_pasta_destino(original, pasta_destino, copiar=True, novo_nome="CODIGO1-P001-00001.tif")
    r2 = arquivar_em_pasta_destino(original, pasta_destino, copiar=True, novo_nome="CODIGO2-P001-00001.tif")

    assert original.name == "Copia de DEST2977.tif"
    assert original.exists()
    assert r1.name == "CODIGO1-P001-00001.tif"
    assert r2.name == "CODIGO2-P001-00001.tif"


def test_pasta_organizada_por_codigo_de_projeto():
    destino = montar_pasta_destino(
        Path("/acervo"), "{ano}/{codigo_projeto_auto} - {projeto}", "1974",
        "Teatro de Santos", codigo_projeto_auto="OCG-P0003",
    )
    assert destino == Path("/acervo/1974/OCG-P0003 - Teatro de Santos")


def test_pasta_sem_codigo_projeto_nao_deixa_separador_solto():
    """Sem projeto identificado nesta prancha (codigo_projeto_auto
    vazio), a pasta cai só no nome de retaguarda, sem um " - " sobrando
    na frente (ex.: "2018/ - Sami Bussab")."""
    destino = montar_pasta_destino(
        Path("/acervo"), "{ano}/{codigo_projeto_auto} - {projeto}", "2018", "Sami Bussab",
    )
    assert destino == Path("/acervo/2018/Sami Bussab")
