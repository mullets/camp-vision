"""Testes unitários da correção inteligente de nomes via similaridade textual."""

import pytest

from database.models import criar_sessao
from database.repository import ConhecimentoRepository


@pytest.fixture
def sessao(tmp_path):
    caminho_db = tmp_path / "teste.sqlite3"
    return criar_sessao(str(caminho_db))


def test_cadastra_novo_arquiteto(sessao):
    repo = ConhecimentoRepository(sessao)
    nome = repo.sugerir_arquiteto("Carlos Barjas Millan")
    assert nome == "Carlos Barjas Millan"


def test_corrige_variacao_de_nome_ja_conhecido(sessao):
    repo = ConhecimentoRepository(sessao)
    repo.sugerir_arquiteto("Carlos Barjas Millan")  # cadastra o nome canônico

    corrigido = repo.sugerir_arquiteto("Carlos Barjas Mlllan")  # variação com erro de OCR
    assert corrigido == "Carlos Barjas Millan"


def test_nome_muito_diferente_nao_e_corrigido(sessao):
    repo = ConhecimentoRepository(sessao)
    repo.sugerir_arquiteto("Carlos Barjas Millan")

    resultado = repo.sugerir_arquiteto("Ana Paula Souza")
    assert resultado == "Ana Paula Souza"


def test_corrige_grafia_de_projeto_ja_conhecido(sessao):
    repo = ConhecimentoRepository(sessao)
    repo.sugerir_projeto("Esporte Clube Sirio")

    corrigido = repo.sugerir_projeto("Esporte Clube Siri0")  # OCR trocou O por 0
    assert corrigido == "Esporte Clube Sirio"


def test_endereco_do_projeto_preenche_a_partir_de_associacao_conhecida(sessao):
    repo = ConhecimentoRepository(sessao)
    repo.registrar_endereco_do_projeto("Esporte Clube Sirio", "Av. Indianópolis, 1192")

    assert repo.endereco_do_projeto("Esporte Clube Sirio") == "Av. Indianópolis, 1192"
    # também deve achar por similaridade, não só grafia idêntica
    assert repo.endereco_do_projeto("Esporte Clube Siri0") == "Av. Indianópolis, 1192"


def test_endereco_do_projeto_sem_associacao_devolve_none(sessao):
    repo = ConhecimentoRepository(sessao)
    repo.sugerir_projeto("Projeto sem endereço cadastrado")

    assert repo.endereco_do_projeto("Projeto sem endereço cadastrado") is None


def test_registrar_endereco_do_projeto_nao_sobrescreve_associacao_existente(sessao):
    repo = ConhecimentoRepository(sessao)
    repo.registrar_endereco_do_projeto("Praça Vila Formosa", "Endereço correto")

    repo.registrar_endereco_do_projeto("Praça Vila Formosa", "Endereço errado (leitura isolada)")

    assert repo.endereco_do_projeto("Praça Vila Formosa") == "Endereço correto"


def test_valor_visto_uma_vez_fica_com_contagem_um(sessao):
    from database.models import Arquiteto

    repo = ConhecimentoRepository(sessao)
    repo.sugerir_arquiteto("Oswaldo Correa Goncalves")

    registro = sessao.query(Arquiteto).filter_by(nome="Oswaldo Correa Goncalves").first()
    assert registro.contagem == 1


def test_segunda_leitura_parecida_confirma_a_contagem(sessao):
    from database.models import Arquiteto

    repo = ConhecimentoRepository(sessao)
    repo.sugerir_arquiteto("Oswaldo Correa Goncalves")
    repo.sugerir_arquiteto("Oswaldo Correa Goncalves")  # 2ª leitura idêntica

    registro = sessao.query(Arquiteto).filter_by(nome="Oswaldo Correa Goncalves").first()
    assert registro.contagem == 2


def test_grafia_mais_completa_vence_ao_confirmar(sessao):
    repo = ConhecimentoRepository(sessao)
    repo.sugerir_arquiteto("Carlos B Millan")  # leitura abreviada, 1ª vez

    corrigido = repo.sugerir_arquiteto("Carlos Barjas Millan")  # leitura mais completa, 2ª vez

    assert corrigido == "Carlos Barjas Millan"
    # e a partir daqui, essa é que vira a canônica pra próximas leituras abreviadas
    assert repo.sugerir_arquiteto("Carlos B Millan") == "Carlos Barjas Millan"


def test_grafia_canonica_congela_apos_confiavel(sessao):
    """Reproduz o bug real visto num lote de 1266 pranchas: depois que
    'Oswaldo Correa Goncalves' já tinha sido confirmado dezenas de
    vezes, uma única leitura ruim ('HOSWALDO CORREA GONÇALVES', com H
    espúrio — 1 caractere mais longa) não pode mais sequestrar a
    grafia canônica e passar a "corrigir" as leituras corretas
    seguintes para a forma errada."""
    repo = ConhecimentoRepository(sessao)
    for _ in range(10):
        repo.sugerir_arquiteto("Oswaldo Correa Goncalves")

    resultado = repo.sugerir_arquiteto("HOSWALDO CORREA GONÇALVES")
    assert resultado == "Oswaldo Correa Goncalves"

    # e a canônica continua correta pras leituras seguintes
    assert repo.sugerir_arquiteto("Oswaldo Correa Goncalves") == "Oswaldo Correa Goncalves"
    repo = ConhecimentoRepository(sessao)
    repo.sugerir_escala("Escala 1:100")
    repo.sugerir_escala("Escala 1:100")  # confirma (2ª leitura)

    assert repo.sugerir_escala("Escala 1:l00") == "Escala 1:100"  # OCR trocou 0 por l
