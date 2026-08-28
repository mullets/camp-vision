"""Testes de utils.metadados_exif — em especial o campo Copyright, que
passou a combinar arquiteto + atribuição institucional (configurável)
em vez do nome do cliente."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from utils.metadados_exif import MetadadosParaGravar, _montar_copyright, gravar_metadados


def test_copyright_combina_arquiteto_e_instituicao():
    m = MetadadosParaGravar(
        arquiteto="Joaquim Barretto Arquitetos Associados SA Ltda.",
        cliente="Eduardo Pinheiro Machado",
        atribuicao_instituicao="CAMP - Casa da Arquitetura Moderna Paulista",
    )
    assert _montar_copyright(m) == (
        "Joaquim Barretto Arquitetos Associados SA Ltda. / CAMP - Casa da Arquitetura Moderna Paulista"
    )


def test_copyright_nao_usa_mais_o_cliente():
    m = MetadadosParaGravar(arquiteto="Fulano", cliente="Nome do Cliente", atribuicao_instituicao="CAMP")
    assert "Nome do Cliente" not in _montar_copyright(m)


def test_copyright_sem_arquiteto_usa_so_a_instituicao():
    m = MetadadosParaGravar(arquiteto="", atribuicao_instituicao="CAMP - Casa da Arquitetura Moderna Paulista")
    assert _montar_copyright(m) == "CAMP - Casa da Arquitetura Moderna Paulista"


def test_copyright_sem_instituicao_configurada_usa_so_o_arquiteto():
    m = MetadadosParaGravar(arquiteto="Fulano de Tal", atribuicao_instituicao="")
    assert _montar_copyright(m) == "Fulano de Tal"


def test_argumentos_reais_passados_ao_exiftool_tem_copyright_correto():
    """Confirma ponta a ponta: os argumentos de linha de comando que
    seriam de fato passados ao exiftool trazem o Copyright/Rights
    montados corretamente, não mais o cliente."""
    import utils.metadados_exif as me

    with patch.object(me, "exiftool_disponivel", return_value=True), \
         patch.object(me, "_CAMINHO_EXIFTOOL", "/usr/bin/exiftool"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        metadados = MetadadosParaGravar(
            projeto="Residência Alphaville",
            cliente="Eduardo Pinheiro Machado",
            arquiteto="Joaquim Barretto Arquitetos Associados SA Ltda.",
            atribuicao_instituicao="CAMP - Casa da Arquitetura Moderna Paulista",
        )
        gravar_metadados(Path("/tmp/teste.jpg"), metadados)

        argumentos = mock_run.call_args[0][0]
        copyright_arg = next(a for a in argumentos if a.startswith("-Copyright="))
        rights_arg = next(a for a in argumentos if a.startswith("-XMP-dc:Rights="))

        esperado = (
            "-Copyright=Joaquim Barretto Arquitetos Associados SA Ltda. "
            "/ CAMP - Casa da Arquitetura Moderna Paulista"
        )
        assert copyright_arg == esperado
        assert rights_arg == esperado.replace("-Copyright=", "-XMP-dc:Rights=")
