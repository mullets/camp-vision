"""
utils/logger.py
================
Sistema de logs do CAMP Vision.

Cria um logger estruturado que grava simultaneamente:
  - em arquivo (logs/campvision.log), com rotação por tamanho;
  - no console;
  - em uma fila (Queue) opcional, consumida pela interface gráfica
    para exibir o log em tempo real.

Cada entrada de erro por arquivo processado registra: arquivo, erro,
tempo, módulo e stacktrace, conforme exigido pelo fluxo de
processamento em lote.
"""

from __future__ import annotations

import contextvars
import logging
import logging.handlers
import queue
import traceback
from pathlib import Path
from typing import Optional

# Nome do arquivo sendo processado no momento, POR THREAD — usado para
# marcar automaticamente cada linha de log com o arquivo a que ela
# pertence (ver `ContextoArquivoFilter` e `arquivo_em_processamento`).
# Necessário porque o processamento em lote roda vários arquivos em
# paralelo (ThreadPoolExecutor): sem isso, mensagens intermediárias de
# detecção de carimbo (ex. "Melhor candidato por conteúdo...") ficam
# intercaladas no log sem indicar a qual arquivo pertencem, tornando
# impossível saber quantas tentativas de orientação de fato rodaram
# para um arquivo específico — precisou ser investigado na prática
# (usuário reportou resultado ruim persistente; não dava pra confirmar
# pelo log se a 2ª estratégia tinha sido tentada em 1 ou 4 orientações
# para aquele arquivo).
_arquivo_atual: contextvars.ContextVar[str] = contextvars.ContextVar("arquivo_atual", default="")


class arquivo_em_processamento:
    """Context manager: marca, para a thread atual, qual arquivo está
    sendo processado — toda mensagem de log emitida dentro do bloco
    `with` ganha automaticamente o prefixo `[nome_do_arquivo]`.

    Uso: `with arquivo_em_processamento(caminho.name): ...`"""

    def __init__(self, nome_arquivo: str):
        self.nome_arquivo = nome_arquivo
        self._token = None

    def __enter__(self):
        self._token = _arquivo_atual.set(self.nome_arquivo)
        return self

    def __exit__(self, *exc_info):
        _arquivo_atual.reset(self._token)
        return False


class ContextoArquivoFilter(logging.Filter):
    """Injeta o arquivo atual (se houver) em todo LogRecord, para os
    formatadores poderem incluí-lo na linha sem que cada chamada de
    log precise passar isso explicitamente."""

    def filter(self, record: logging.LogRecord) -> bool:
        nome = _arquivo_atual.get()
        record.arquivo_contexto = f"[{nome}] " if nome else ""
        return True


class QueueLogHandler(logging.Handler):
    """Handler que publica registros de log em uma Queue thread-safe,
    permitindo que a interface gráfica exiba o log em tempo real sem
    acoplamento direto com o restante do sistema."""

    def __init__(self, log_queue: "queue.Queue[str]"):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.log_queue.put_nowait(msg)
        except Exception:
            pass  # Nunca deixar o logging derrubar o processamento.


def configurar_logging(log_dir: Path, gui_queue: Optional["queue.Queue[str]"] = None) -> logging.Logger:
    """Configura o logger raiz 'campvision' com handlers de arquivo,
    console e (opcionalmente) fila para a interface gráfica."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("campvision")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formato = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(arquivo_contexto)s%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # O filtro vai nos HANDLERS, não no logger raiz: a maioria dos
    # módulos usa loggers FILHOS (ex. "campvision.scanner.detector_carimbo"),
    # e um filtro adicionado só ao logger "campvision" não se aplica a
    # registros originados em loggers filhos — só filtros de handler
    # rodam de forma uniforme, não importa qual logger originou o
    # registro.
    arquivo_handler = logging.handlers.RotatingFileHandler(
        log_dir / "campvision.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    arquivo_handler.setFormatter(formato)
    arquivo_handler.setLevel(logging.DEBUG)
    arquivo_handler.addFilter(ContextoArquivoFilter())
    logger.addHandler(arquivo_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formato)
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(ContextoArquivoFilter())
    logger.addHandler(console_handler)

    if gui_queue is not None:
        queue_handler = QueueLogHandler(gui_queue)
        queue_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(arquivo_contexto)s%(message)s",
                                                       datefmt="%H:%M:%S"))
        queue_handler.setLevel(logging.INFO)
        queue_handler.addFilter(ContextoArquivoFilter())
        logger.addHandler(queue_handler)

    return logger


def registrar_erro_processamento(logger: logging.Logger, arquivo: str, modulo: str, excecao: Exception) -> None:
    """Registra um erro ocorrido durante o processamento de um arquivo
    específico, incluindo stacktrace completo, conforme especificação
    do sistema de logs."""
    stacktrace = "".join(traceback.format_exception(type(excecao), excecao, excecao.__traceback__))
    logger.error(
        "Falha ao processar arquivo=%s | modulo=%s | erro=%s\n%s",
        arquivo, modulo, str(excecao), stacktrace,
    )
