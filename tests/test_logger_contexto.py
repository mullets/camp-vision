"""Testes de utils.logger — o contexto de "arquivo atual" que marca
automaticamente cada linha de log com o nome do arquivo sendo
processado, mesmo em loggers filhos e com múltiplas threads rodando
arquivos diferentes ao mesmo tempo (o padrão real do processamento em
lote). Sem isso, mensagens intermediárias de detecção de carimbo
ficavam impossíveis de atribuir a um arquivo específico num lote com
várias threads — precisou ser investigado na prática."""

import logging
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from utils.logger import configurar_logging, arquivo_em_processamento


def test_contexto_de_arquivo_aparece_na_linha_de_log():
    log_dir = Path(tempfile.mkdtemp())
    configurar_logging(log_dir)
    logger_filho = logging.getLogger("campvision.teste.modulo_filho")

    with arquivo_em_processamento("PRANCHA_X.tif"):
        logger_filho.info("mensagem de teste")

    conteudo = (log_dir / "campvision.log").read_text()
    assert "[PRANCHA_X.tif] mensagem de teste" in conteudo


def test_sem_contexto_nao_aparece_prefixo_vazio():
    log_dir = Path(tempfile.mkdtemp())
    configurar_logging(log_dir)
    logger_raiz = logging.getLogger("campvision")

    logger_raiz.info("mensagem fora de qualquer arquivo")

    conteudo = (log_dir / "campvision.log").read_text()
    assert "mensagem fora de qualquer arquivo" in conteudo
    assert "[]" not in conteudo


def test_contexto_correto_com_varias_threads_processando_arquivos_diferentes():
    """O cenário real: ThreadPoolExecutor processando vários arquivos
    ao mesmo tempo. Cada thread precisa ver SÓ o nome do arquivo que
    ela mesma está processando, mesmo com logging vindo de um módulo
    filho (ex. campvision.scanner.detector_carimbo)."""
    log_dir = Path(tempfile.mkdtemp())
    configurar_logging(log_dir)
    logger_filho = logging.getLogger("campvision.scanner.detector_carimbo")

    def processar(nome):
        with arquivo_em_processamento(nome):
            logger_filho.info("tentativa de deteccao")
            time.sleep(0.02)
            logger_filho.info("segunda tentativa")

    nomes = [f"ARQUIVO_{i}.tif" for i in range(5)]
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(processar, nomes))

    conteudo = (log_dir / "campvision.log").read_text()
    for nome in nomes:
        assert conteudo.count(f"[{nome}] tentativa de deteccao") == 1
        assert conteudo.count(f"[{nome}] segunda tentativa") == 1


def test_contexto_e_restaurado_ao_sair_do_bloco():
    log_dir = Path(tempfile.mkdtemp())
    configurar_logging(log_dir)
    logger_raiz = logging.getLogger("campvision")

    with arquivo_em_processamento("PRANCHA_Y.tif"):
        logger_raiz.info("dentro do bloco")
    logger_raiz.info("fora do bloco")

    conteudo = (log_dir / "campvision.log").read_text()
    assert "[PRANCHA_Y.tif] dentro do bloco" in conteudo
    assert "[]" not in conteudo
    linha_fora = [l for l in conteudo.splitlines() if "fora do bloco" in l][0]
    assert "PRANCHA_Y.tif" not in linha_fora
