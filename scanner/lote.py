"""
scanner/lote.py
================
Orquestra o processamento em lote de milhares de arquivos TIFF em uma
pasta, com paralelismo via ThreadPoolExecutor (o gargalo é I/O,
chamadas ao binário do Tesseract e chamadas de rede para a IA — por
isso threads, e não processos, evitando o alto custo de serializar
imagens grandes entre processos).

Suporta cancelamento cooperativo (via threading.Event) e notifica o
progresso através de um callback, permitindo que a interface gráfica
atualize barra de progresso, contagem de arquivos e ETA em tempo
real.
"""

from __future__ import annotations

import logging
import re
import shutil
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2

from ai.interpretador import InterpretadorIA
from classificacao.classificador import criar_classificador
from database.models import criar_sessao
from exportacao.exportador import Exportador
from ocr.motor import criar_motor_ocr
from scanner.detector_carimbo import criar_detector
from scanner.pipeline import PipelineProcessamento, ResultadoProcessamento
from scanner.propagacao import propagar_metadados_projeto, unificar_grafias
from utils.texto import normalizar_maiusculas
from utils.renomeador import GeradorSequencial, parece_ja_renomeado, sugerir_prefixo_projeto
from utils.tempo import EstimadorTempo
from utils.logger import arquivo_em_processamento

logger = logging.getLogger("campvision.lote")


@dataclass
class ProgressoLote:
    total: int
    concluidos: int = 0
    sucesso: int = 0
    falhas: int = 0
    tempo_restante_formatado: str = ""
    arquivo_atual: str = ""


@dataclass
class ConfiguracaoLote:
    pasta_entrada: Path
    pasta_saida: Path
    formatos_aceitos: list[str]
    quantidade_threads: int
    idiomas_ocr: list[str]
    tamanho_miniatura: int
    qualidade_miniatura: int
    salvar_miniaturas: bool
    salvar_carimbos: bool
    ia_api_key: str
    ia_modelo: str
    ia_habilitada: bool
    caminho_db: str
    ocr_motor: str = "tesseract"
    deteccao_carimbo_modo: str = "heuristico"
    caminho_modelo_carimbo: str = ""
    carimbo_regiao_busca: str = "automatico"
    classificacao_modo: str = "regras"
    caminho_modelo_classificacao: str = ""
    confianca_minima_ml: float = 0.5
    tamanho_imagem_ml: int = 960
    codigo_projeto: str = ""
    renomeacao_habilitada: bool = True
    renomeacao_padrao: str = "{codigo_projeto_auto}-{sequencial_no_projeto} - {projeto} - {nome_prancha}"
    renomeacao_digitos_sequencial: int = 5
    gravar_metadados_exif: bool = True
    arquivamento_habilitado: bool = True
    arquivamento_padrao_pastas: str = "{ano}/{codigo_projeto_auto} - {projeto}"
    arquivamento_pasta_raiz: Optional[Path] = None
    arquivamento_copiar: bool = True
    deteccao_multiorientacao: bool = True
    nome_projeto: str = ""
    atribuicao_instituicao: str = ""


def _atribuir_codigos_por_projeto(analises_por_arquivo: dict) -> None:
    """Numera os projetos do lote e as pranchas dentro de cada um.

    Cada código vem com um PREFIXO tirado automaticamente do
    arquiteto/escritório (ou, na ausência dele, da cidade) lido no
    carimbo — ex.: "OCG-P0001" para um projeto de "Oswaldo Correa
    Goncalves" — para que o código já identifique de qual acervo se
    trata, sem precisar de configuração manual. Roda DEPOIS da
    propagação e da unificação de grafias (ver scanner/lote.executar),
    então o arquiteto já está completo/unificado em todas as pranchas
    do mesmo projeto, garantindo um prefixo consistente dentro do
    grupo.

    Pranchas sem projeto identificado caem num grupo próprio, para não
    serem misturadas com as de um projeto conhecido.

    Também calcula, por grupo de projeto, um único `ano_pasta` (o ano
    mais comum entre as pranchas do grupo) para uso exclusivo na PASTA
    de arquivamento — evitando que pranchas do mesmo projeto com anos
    individualmente distintos (datação de detalhe posterior, "Ano
    desconhecido" só nalgumas, etc.) fragmentem um único projeto em
    várias pastas de ano diferentes. O `ano` de cada prancha (usado no
    CSV/EXIF) não é alterado por isso."""
    codigos: dict[str, str] = {}
    contadores: dict[str, int] = {}
    metadados_por_chave: dict[str, list] = {}

    for caminho in sorted(analises_por_arquivo, key=lambda c: _chave_natural(c.name)):
        analises = analises_por_arquivo[caminho]
        if not analises:
            continue
        metadados = analises[0].metadados
        chave = normalizar_maiusculas(metadados.projeto).strip() or "SEM PROJETO"

        if chave not in codigos:
            prefixo = sugerir_prefixo_projeto(metadados.arquiteto, metadados.cidade)
            numero_projeto = f"P{len(codigos) + 1:04d}"
            codigos[chave] = f"{prefixo}-{numero_projeto}" if prefixo else numero_projeto
            contadores[chave] = 0
        contadores[chave] += 1

        metadados.codigo_projeto_auto = codigos[chave]
        metadados.sequencial_no_projeto = contadores[chave]
        metadados_por_chave.setdefault(chave, []).append(metadados)

    for grupo_metadados in metadados_por_chave.values():
        anos = [m.ano.strip() for m in grupo_metadados if m.ano and m.ano.strip()]
        ano_comum = Counter(anos).most_common(1)[0][0] if anos else ""
        for metadados in grupo_metadados:
            metadados.ano_pasta = ano_comum


def _chave_natural(nome: str) -> tuple:
    """Ordena tratando números como números (DEST9 antes de DEST10)."""
    return tuple(int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", nome))


class ProcessadorLote:
    """Executa o processamento paralelo de todos os TIFFs de uma
    pasta, expondo pontos de cancelamento e callback de progresso."""

    def __init__(self, config: ConfiguracaoLote, callback_progresso: Optional[Callable[[ProgressoLote], None]] = None):
        self.config = config
        self.callback_progresso = callback_progresso
        self.evento_cancelamento = threading.Event()
        self._lock = threading.Lock()

    def cancelar(self) -> None:
        logger.info("Cancelamento solicitado pelo usuário.")
        self.evento_cancelamento.set()

    def resolver_raiz_arquivamento(self) -> Path:
        """Calcula a raiz da estrutura de arquivamento (ano/projeto) UMA
        ÚNICA VEZ por execução, sempre como um caminho fixo — nunca
        derivado da localização atual de um arquivo (isso é o que
        causava aninhamento infinito de pastas em reprocessamentos).

        Por padrão (quando não configurada explicitamente), a raiz é
        uma pasta IRMÃ da pasta selecionada — nunca dentro dela — para
        que os arquivos já arquivados nunca sejam encontrados de novo
        por `listar_arquivos` numa próxima execução."""
        if self.config.arquivamento_pasta_raiz is not None:
            return self.config.arquivamento_pasta_raiz
        return self.config.pasta_entrada.parent / f"{self.config.pasta_entrada.name}_catalogado"

    def listar_arquivos(self, raiz_arquivamento: Optional[Path] = None) -> list[Path]:
        extensoes = {e.lower() for e in self.config.formatos_aceitos}
        pastas_excluidas = {self.config.pasta_saida.resolve()}
        if raiz_arquivamento is not None:
            pastas_excluidas.add(raiz_arquivamento.resolve())

        arquivos = []
        ignorados_ja_processados = 0
        for p in sorted(self.config.pasta_entrada.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in extensoes:
                continue
            # Nunca reprocessar arquivos que já estão dentro da pasta de
            # saída ou da estrutura de arquivamento (proteção contra
            # reprocessamento acidental ao rodar o lote de novo).
            if any(pasta in p.resolve().parents for pasta in pastas_excluidas):
                continue
            # Proteção redundante: mesmo fora dessas pastas, um arquivo
            # cujo nome já parece ter sido gerado pelo CAMP Vision para
            # este mesmo código de projeto é pulado, com aviso.
            if parece_ja_renomeado(p.stem, self.config.codigo_projeto, self.config.renomeacao_digitos_sequencial):
                ignorados_ja_processados += 1
                continue
            arquivos.append(p)

        if ignorados_ja_processados:
            logger.warning(
                "%d arquivo(s) ignorado(s) por já parecerem processados (nome no padrão '%s-...-NNNNN'). "
                "Se forem arquivos novos, use um código de projeto diferente.",
                ignorados_ja_processados, self.config.codigo_projeto,
            )
        return arquivos

    def _verificar_espaco_em_disco(self, arquivos: list[Path], raiz_arquivamento: Path) -> None:
        """Confere, ANTES de começar, se há espaço suficiente na pasta
        de arquivamento para copiar todos os arquivos do lote.

        Achado num caso real: o disco encheu no meio de um lote de 147
        arquivos, e a falta de espaço derrubou em cascata não só a
        cópia dos arquivos (`[Errno 28] No space left on device`) mas
        também o banco de conhecimento SQLite, que também grava em
        disco (`sqlite3.OperationalError: unable to open database
        file`) — o usuário só descobriu depois de já ter esperado o
        lote inteiro "terminar" com quase tudo em erro.

        O processamento em lote copia cada arquivo original para a
        pasta de arquivamento (mantendo o original), então por um
        tempo é preciso espaço para praticamente DUAS cópias de tudo.
        Só avisa (log de warning) — não interrompe o processamento,
        porque é uma estimativa: parte dos arquivos pode já estar
        arquivada de uma execução anterior, por exemplo."""
        try:
            tamanho_total = sum(a.stat().st_size for a in arquivos if a.exists())
            raiz_arquivamento.mkdir(parents=True, exist_ok=True)
            espaco_livre = shutil.disk_usage(raiz_arquivamento).free
        except OSError as exc:
            logger.debug("Não foi possível checar espaço em disco antes de iniciar: %s", exc)
            return

        # Margem de segurança: miniaturas, recortes de carimbo e o
        # banco de conhecimento também usam disco, além do tamanho
        # bruto dos arquivos originais copiados.
        margem = 1.10
        necessario = tamanho_total * margem
        if espaco_livre < necessario:
            logger.warning(
                "Espaço em disco pode não ser suficiente para este lote: são necessários "
                "~%.1f GB para copiar os %d arquivo(s) para a pasta de arquivamento, mas só "
                "há ~%.1f GB livres em '%s'. O processamento vai continuar, mas pode falhar "
                "no meio (erro 'No space left on device') se o espaço acabar antes de terminar.",
                necessario / 1e9, len(arquivos), espaco_livre / 1e9, raiz_arquivamento,
            )

    def executar(self) -> list[ResultadoProcessamento]:
        # IMPORTANTE — evita "oversubscription" de CPU: por padrão, o
        # PyTorch (modelo treinado de carimbo/classificação) usa TODOS
        # os núcleos da máquina em CADA chamada individual (paralelismo
        # interno via OpenMP/MKL). Isso é ótimo quando só há UMA
        # chamada por vez — mas aqui já paralelizamos por FORA, com
        # `quantidade_threads` threads chamando o modelo ao mesmo
        # tempo. Sem este ajuste, N threads brigam pelos mesmos
        # núcleos ao mesmo tempo (mais troca de contexto que trabalho
        # útil), e o lote fica MAIS LENTO com mais threads, não mais
        # rápido — efeito observado na prática: lote de ~150 arquivos
        # levando mais de 6h. Limitando cada chamada a 1 núcleo, as
        # `quantidade_threads` chamadas concorrentes de fato rodam uma
        # por núcleo, em paralelo de verdade. Import protegido: torch
        # só está instalado quando algum modelo treinado é usado.
        try:
            import torch
            torch.set_num_threads(1)
        except ImportError:
            pass

        cv2.setNumThreads(1)

        raiz_arquivamento = self.resolver_raiz_arquivamento() if self.config.arquivamento_habilitado else None
        arquivos = self.listar_arquivos(raiz_arquivamento)
        total = len(arquivos)
        logger.info("Iniciando processamento em lote: %d arquivos encontrados em %s", total, self.config.pasta_entrada)

        if total == 0:
            logger.warning("Nenhum arquivo compatível encontrado em %s", self.config.pasta_entrada)
            return []

        if raiz_arquivamento is not None:
            self._verificar_espaco_em_disco(arquivos, raiz_arquivamento)

        exportador = Exportador(self.config.pasta_saida)
        interpretador_ia = InterpretadorIA(
            api_key=self.config.ia_api_key,
            modelo=self.config.ia_modelo,
            habilitada=self.config.ia_habilitada,
        )

        # Os modelos treinados (se configurados) são carregados uma
        # única vez aqui e reutilizados por todas as threads do lote —
        # tanto para não pagar o custo de carregamento a cada arquivo,
        # quanto porque a inferência em modo eval (sem atualização de
        # pesos) é segura para chamadas concorrentes. Em caso de falha
        # no carregamento, as fábricas já caem automaticamente para a
        # estratégia heurística/por regras (ver logs).
        ocr_fn = criar_motor_ocr(self.config.ocr_motor)

        if self.config.ocr_motor == "tesseract":
            # Confirma a disponibilidade do tesseract UMA VEZ, antes de
            # abrir as threads do lote. Checar isso de forma concorrente
            # (várias threads chamando ao mesmo tempo, cada uma
            # disparando sua própria checagem de PATH) mostrou-se
            # sujeito a falso-negativo ocasional em testes reais — uma
            # vez confirmado aqui, o resultado positivo fica em cache
            # (ver ocr/tesseract_ocr.tesseract_disponivel) e nenhuma
            # thread precisa checar de novo durante o processamento.
            from ocr.tesseract_ocr import tesseract_disponivel
            tesseract_disponivel()

        # O detector recebe o motor de OCR para, no modo heurístico
        # com busca automática, verificar por CONTEÚDO qual região
        # candidata é de fato o carimbo (essencial quando a posição
        # varia livremente entre pranchas — ver
        # scanner/detector_carimbo.detectar_carimbo_verificado).
        detector_carimbo_fn = criar_detector(
            modo=self.config.deteccao_carimbo_modo,
            caminho_modelo=self.config.caminho_modelo_carimbo,
            confianca_minima=self.config.confianca_minima_ml,
            regiao_fixa=self.config.carimbo_regiao_busca,
            ocr_fn=ocr_fn,
            idiomas_ocr=self.config.idiomas_ocr,
            tamanho_imagem_ml=self.config.tamanho_imagem_ml,
        )
        # 2ª estratégia de detecção (ver
        # scanner/pipeline.py: _tentar_fallback_geometrico): só faz
        # sentido quando o detector PRIMÁRIO é o modelo treinado — a
        # busca geométrica com verificação por conteúdo entra como
        # complemento genuíno quando o modelo não acha nada, em vez de
        # simplesmente desistir da prancha. Quando o primário já É a
        # heurística, não há fallback nenhum pra montar (tentar de novo
        # a mesma estratégia não ajudaria e só custaria tempo à toa).
        detector_carimbo_fallback_fn = None
        if self.config.deteccao_carimbo_modo == "modelo_treinado":
            detector_carimbo_fallback_fn = criar_detector(
                modo="heuristico",
                regiao_fixa=self.config.carimbo_regiao_busca,
                ocr_fn=ocr_fn,
                idiomas_ocr=self.config.idiomas_ocr,
            )
        classificador_fn = criar_classificador(
            modo=self.config.classificacao_modo,
            caminho_modelo=self.config.caminho_modelo_classificacao,
        )

        # Gerador de códigos sequenciais únicos para a renomeação dos
        # arquivos originais — uma única instância compartilhada entre
        # todas as threads do lote (thread-safe internamente), e
        # persistida no banco para nunca colidir entre execuções.
        gerador_sequencial = GeradorSequencial(self.config.caminho_db) if self.config.renomeacao_habilitada else None

        estimador = EstimadorTempo(total_itens=total)
        progresso = ProgressoLote(total=total)
        resultados: list[ResultadoProcessamento] = []

        def montar_pipeline():
            # Cada thread tem sua própria sessão de banco (SQLAlchemy
            # não é thread-safe por sessão compartilhada).
            sessao = criar_sessao(self.config.caminho_db)
            return sessao, PipelineProcessamento(
                interpretador_ia=interpretador_ia,
                exportador=exportador,
                sessao_db=sessao,
                idiomas_ocr=self.config.idiomas_ocr,
                tamanho_miniatura=self.config.tamanho_miniatura,
                qualidade_miniatura=self.config.qualidade_miniatura,
                salvar_miniaturas=self.config.salvar_miniaturas,
                salvar_carimbos=self.config.salvar_carimbos,
                detector_carimbo_fn=detector_carimbo_fn,
                detector_carimbo_fallback_fn=detector_carimbo_fallback_fn,
                classificador_fn=classificador_fn,
                ocr_fn=ocr_fn,
                codigo_projeto=self.config.codigo_projeto,
                gerador_sequencial=gerador_sequencial,
                renomeacao_habilitada=self.config.renomeacao_habilitada,
                renomeacao_padrao=self.config.renomeacao_padrao,
                renomeacao_digitos_sequencial=self.config.renomeacao_digitos_sequencial,
                gravar_exif=self.config.gravar_metadados_exif,
                arquivamento_habilitado=self.config.arquivamento_habilitado,
                arquivamento_padrao_pastas=self.config.arquivamento_padrao_pastas,
                arquivamento_pasta_raiz=raiz_arquivamento,
                arquivamento_copiar=self.config.arquivamento_copiar,
                # Só vale para o detector TREINADO: ele não é invariante
                # a rotação. A heurística já varre a prancha inteira por
                # regiões e pontua por conteúdo, então girar a imagem só
                # multiplicaria o custo (cada detecção heurística roda OCR
                # em várias regiões) sem ganho.
                deteccao_multiorientacao=(
                    self.config.deteccao_multiorientacao
                    and self.config.deteccao_carimbo_modo == "modelo_treinado"
                ),
                nome_projeto=self.config.nome_projeto or self.config.pasta_entrada.name,
                atribuicao_instituicao=self.config.atribuicao_instituicao,
            )

        def analisar_um(caminho: Path):
            """FASE 1 — só analisa (não renomeia nem move nada)."""
            sessao, pipeline = montar_pipeline()
            try:
                with arquivo_em_processamento(caminho.name):
                    return pipeline.analisar_arquivo(caminho)
            finally:
                sessao.close()

        def finalizar_um(caminho: Path, analises) -> ResultadoProcessamento:
            """FASE 2 — renomeia, arquiva e persiste, já com os
            metadados propagados entre as pranchas do lote."""
            sessao, pipeline = montar_pipeline()
            try:
                with arquivo_em_processamento(caminho.name):
                    return pipeline.finalizar_arquivo(caminho, analises)
            finally:
                sessao.close()

        analises_por_arquivo: dict[Path, list] = {}
        try:
            # ---- FASE 1: analisar TODAS as pranchas (nada é alterado em disco) ----
            logger.info("Fase 1/2 — analisando %d prancha(s) (leitura, OCR e detecção de carimbo)...", total)
            with ThreadPoolExecutor(max_workers=self.config.quantidade_threads) as executor:
                futuros = {}
                for caminho in arquivos:
                    if self.evento_cancelamento.is_set():
                        break
                    futuros[executor.submit(analisar_um, caminho)] = caminho

                for futuro in as_completed(futuros):
                    caminho = futuros[futuro]
                    if self.evento_cancelamento.is_set():
                        logger.info("Processamento cancelado durante a análise. %d/%d arquivos.",
                                    progresso.concluidos, total)
                        # IMPORTANTE: sem isto, o `with` abaixo esperava
                        # (shutdown padrão é wait=True) TODOS os arquivos
                        # já enfileirados terminarem de processar — e
                        # como o laço de submissão logo acima é quase
                        # instantâneo (executor.submit só enfileira, não
                        # espera nada), na prática quase o lote inteiro
                        # já tinha sido enfileirado antes do clique em
                        # "Cancelar". O resultado: cancelar não
                        # interrompia nada de verdade, só parava de
                        # atualizar a barra de progresso — o
                        # processamento continuava rodando em segundo
                        # plano até esgotar a fila inteira.
                        # `cancel_futures=True` cancela os que ainda não
                        # começaram a rodar; os poucos já em andamento
                        # (até `quantidade_threads` deles) terminam o
                        # arquivo atual normalmente, mas nada além disso.
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    falhou = False
                    try:
                        analises_por_arquivo[caminho] = futuro.result()
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Erro inesperado analisando %s: %s", caminho, exc)
                        resultados.append(ResultadoProcessamento(arquivo=caminho, sucesso=False, erro=str(exc)))
                        falhou = True

                    # A ANÁLISE é a etapa lenta (leitura da imagem, OCR,
                    # detecção de carimbo) — é ela que precisa alimentar
                    # a barra de progresso. A finalização (renomear e
                    # copiar) é rápida e roda depois, sobre dados já
                    # prontos.
                    with self._lock:
                        estimador.registrar_item_concluido()
                        progresso.concluidos += 1
                        if falhou:
                            progresso.falhas += 1
                        progresso.arquivo_atual = caminho.name
                        progresso.tempo_restante_formatado = estimador.eta_formatado
                    if self.callback_progresso:
                        self.callback_progresso(progresso)

            # ---- Propagação entre pranchas, ANTES de nomear/arquivar ----
            # Precisa acontecer aqui, entre as duas fases: é o que
            # permite que o ano (e demais campos) descoberto em UMA
            # prancha defina o nome e a pasta de arquivamento das
            # OUTRAS do mesmo projeto. Enquanto a propagação rodava só
            # no fim, ela corrigia o CSV mas chegava tarde demais para
            # o nome do arquivo e para a pasta "Ano desconhecido".
            metadados_do_lote = []
            for caminho, analises in analises_por_arquivo.items():
                if not analises:
                    continue
                metadados = analises[0].metadados
                # A propagação ordena pelo nome original do arquivo
                # (ordem de digitalização) e registra o que inferiu.
                metadados.arquivo = caminho.name
                metadados.arquivo_original = caminho.name
                metadados_do_lote.append(metadados)
            try:
                propagar_metadados_projeto(metadados_do_lote)
            except Exception as exc:  # noqa: BLE001
                # Propagação é um refinamento: se falhar, as pranchas
                # ainda devem ser renomeadas, arquivadas e exportadas
                # com o que foi lido individualmente.
                logger.error("Falha ao propagar metadados entre as pranchas (%s). Seguindo sem propagação.", exc)

            # Unifica grafias divergentes ANTES de numerar projetos e
            # montar pastas: sem isto, "CIA. TELEFONICA- GUARUJA" e
            # "CIA. TELEFONICA - GUARUVA" viram dois projetos e duas
            # pastas para o mesmo acervo.
            try:
                unificar_grafias(metadados_do_lote)
            except Exception as exc:  # noqa: BLE001
                logger.error("Falha ao unificar grafias (%s). Seguindo com os valores como lidos.", exc)

            # ---- Código por projeto (P0001, P0002...) ----
            # Só pode ser feito AQUI: depende do nome do projeto de
            # cada prancha, que a propagação acabou de completar. Cada
            # projeto distinto do lote recebe um código na ordem em que
            # aparece na digitalização, e cada prancha recebe sua
            # posição DENTRO daquele projeto — de modo que o nome final
            # (P0001-001) identifique projeto e folha sem depender do
            # nome original do arquivo.
            _atribuir_codigos_por_projeto(analises_por_arquivo)

            # ---- FASE 2: renomear, arquivar e persistir ----
            logger.info("Fase 2/2 — renomeando e arquivando %d prancha(s)...", len(analises_por_arquivo))
            with ThreadPoolExecutor(max_workers=self.config.quantidade_threads) as executor:
                futuros = {
                    executor.submit(finalizar_um, caminho, analises): caminho
                    for caminho, analises in analises_por_arquivo.items()
                }
                for futuro in as_completed(futuros):
                    caminho = futuros[futuro]
                    if self.evento_cancelamento.is_set():
                        logger.info("Processamento cancelado durante o arquivamento. %d/%d arquivos gravados.",
                                    len(resultados), len(analises_por_arquivo))
                        # Mesmo motivo da Fase 1: sem isto, o `with`
                        # esperaria todo o resto do lote já enfileirado
                        # terminar de renomear/arquivar antes de
                        # devolver o controle.
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    try:
                        resultado = futuro.result()
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Erro inesperado finalizando %s: %s", caminho, exc)
                        resultado = ResultadoProcessamento(arquivo=caminho, sucesso=False, erro=str(exc))

                    resultados.append(resultado)
                    # O progresso (concluidos/ETA) já foi contabilizado
                    # na fase de análise — aqui só registramos o
                    # desfecho e o arquivo sendo gravado, sem contar
                    # duas vezes o mesmo arquivo.
                    with self._lock:
                        progresso.sucesso += 1 if resultado.sucesso else 0
                        if not resultado.sucesso:
                            progresso.falhas += 1
                        progresso.arquivo_atual = f"gravando {resultado.arquivo.name}"

                    if self.callback_progresso:
                        self.callback_progresso(progresso)
        except Exception as exc:  # noqa: BLE001
            # Um erro em qualquer ponto do lote não pode custar o
            # trabalho já feito: sem isto, uma exceção depois da
            # análise abortava o arquivamento E a exportação, e o
            # usuário terminava com os recortes de carimbo na pasta e
            # mais nada — nenhum CSV, nenhuma prancha arquivada.
            logger.exception("Erro durante o processamento do lote (%s). Exportando o que já foi processado.", exc)
        finally:
            if gerador_sequencial is not None:
                gerador_sequencial.fechar()

        # Exporta os arquivos finais (CSV/XLSX/JSON) mesmo se o lote foi
        # cancelado ou interrompido por erro, para não perder o
        # trabalho já realizado.
        try:
            exportador.exportar_tudo()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Falha ao exportar a catalogação (%s).", exc)
        logger.info("Processamento em lote finalizado: %d sucesso, %d falhas.",
                    progresso.sucesso, progresso.falhas)
        return resultados
