"""
scanner/pipeline.py
====================
Orquestra o fluxo completo de processamento de cada arquivo TIFF,
conforme a especificação do projeto:

  Abrir imagem -> verificar resolução -> corrigir orientação ->
  melhorar contraste -> reduzir ruído -> localizar carimbo ->
  recortar carimbo -> OCR -> interpretação por IA -> extrair
  metadados -> classificar -> renomear arquivo original com código
  único de arquivamento -> gravar metadados no arquivo (EXIF/IPTC) ->
  gerar miniatura -> salvar resultados.

Para arquivos multipágina, a extração de metadados/renomeação é
decidida a partir da primeira página (a mais representativa da
prancha); as páginas seguintes reaproveitam o mesmo arquivo já
renomeado.

O pipeline é desenhado para nunca interromper o lote inteiro por
causa de um único arquivo com problema: qualquer exceção em uma
etapa é capturada, registrada no log (com stacktrace) e no próprio
registro de exportação (campo de erro), e o processamento segue para
o próximo arquivo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ai.fallback_regras import ano_valido
from ai.interpretador import InterpretadorIA, MetadadosPrancha
from classificacao.classificador import ResultadoClassificacao, classificar as classificar_por_regras
from database.models import Prancha
from database.repository import ConhecimentoRepository, PranchaRepository
from exportacao.exportador import Exportador, RegistroExportacao
from ocr.base import ResultadoOCR
from ocr.tesseract_ocr import extrair_texto as extrair_texto_padrao
from scanner.detector_carimbo import CarimboDetectado, detectar_carimbo
import cv2

from scanner.detector_carimbo import _contar_acertos_aproximados, corrigir_orientacao_do_carimbo
from scanner.folha_visual import ler_numero_folha
from scanner.leitor_imagem import (
    MARGEM_RECORTE_CARIMBO, ImagemCarregada, _rotacionar, carregar_paginas, corrigir_orientacao,
    realcar_para_ocr, reduzir_para_ocr,
)
from utils.texto import normalizar_maiusculas
from utils.arquivamento import arquivar_em_pasta_destino, montar_pasta_destino
from utils.logger import registrar_erro_processamento
from utils.metadados_exif import MetadadosParaGravar, gravar_metadados
from utils.renomeador import GeradorSequencial, montar_nome_arquivo, montar_nome_prancha, parece_ja_renomeado

logger = logging.getLogger("campvision.pipeline")

# Acima desta confiança a detecção já é boa o bastante para não valer
# testar as orientações espelhadas (que custam mais quatro inferências).
CONFIANCA_BOA_O_BASTANTE = 0.60

# Lado máximo da cópia usada para DESCOBRIR a orientação do carimbo.
# Precisa ser pequeno: são até 8 cópias por página, e pranchas de
# arquitetura chegam a 12000x9000 px (324 MB cada em memória).
TAMANHO_BUSCA_ORIENTACAO = 1600


@dataclass
class ResultadoProcessamento:
    arquivo: Path
    sucesso: bool
    erro: Optional[str] = None


@dataclass
class _AnalisePagina:
    """Resultado intermediário da análise de uma página, antes da
    decisão de renomeação e da persistência final."""

    # Deliberadamente NÃO guarda a imagem: o lote analisa TODOS os
    # arquivos antes de renomear/arquivar qualquer um (para que a
    # propagação de metadados entre pranchas já esteja feita na hora
    # de decidir nome e pasta), e segurar as imagens de um acervo
    # inteiro em memória estouraria a RAM com pranchas de 12000px.
    indice_pagina: int
    texto_ocr: str
    confianca_ocr: float
    metadados: MetadadosPrancha
    classificacao: ResultadoClassificacao
    caminho_carimbo: Optional[Path]
    caminho_miniatura: Optional[Path] = None


class PipelineProcessamento:
    """Processa um único arquivo TIFF (podendo conter várias páginas),
    executando todas as etapas do fluxo e persistindo os resultados
    no banco de dados e no exportador."""

    def __init__(
        self,
        interpretador_ia: InterpretadorIA,
        exportador: Exportador,
        sessao_db,
        idiomas_ocr: list[str],
        tamanho_miniatura: int,
        qualidade_miniatura: int,
        salvar_miniaturas: bool,
        salvar_carimbos: bool,
        detector_carimbo_fn: Optional[Callable[[np.ndarray], Optional[CarimboDetectado]]] = None,
        detector_carimbo_fallback_fn: Optional[Callable[[np.ndarray], Optional[CarimboDetectado]]] = None,
        classificador_fn: Optional[Callable[..., ResultadoClassificacao]] = None,
        ocr_fn: Optional[Callable[[np.ndarray, list], ResultadoOCR]] = None,
        codigo_projeto: str = "",
        gerador_sequencial: Optional[GeradorSequencial] = None,
        renomeacao_habilitada: bool = True,
        renomeacao_padrao: str = "{codigo_projeto}-{prancha}-{sequencial}",
        renomeacao_digitos_sequencial: int = 5,
        gravar_exif: bool = True,
        arquivamento_habilitado: bool = True,
        arquivamento_padrao_pastas: str = "{ano}/{projeto}",
        arquivamento_pasta_raiz: Optional[Path] = None,
        arquivamento_copiar: bool = True,
        deteccao_multiorientacao: bool = True,
        nome_projeto: str = "",
        atribuicao_instituicao: str = "",
    ):
        self.interpretador_ia = interpretador_ia
        self.exportador = exportador
        self.conhecimento = ConhecimentoRepository(sessao_db)
        self.pranchas_repo = PranchaRepository(sessao_db)
        self.idiomas_ocr = idiomas_ocr
        self.tamanho_miniatura = tamanho_miniatura
        self.qualidade_miniatura = qualidade_miniatura
        self.salvar_miniaturas = salvar_miniaturas
        self.salvar_carimbos = salvar_carimbos
        # Estratégias plugáveis: por padrão, heurística (OpenCV) e
        # classificação por regras — ver scanner/detector_carimbo.py e
        # classificacao/classificador.py para as fábricas que também
        # aceitam modelos treinados (ver pacote ml/).
        self.detector_carimbo_fn = detector_carimbo_fn or detectar_carimbo
        # Segunda estratégia de detecção (ver _tentar_fallback_geometrico),
        # usada só quando o primário não acha nada. Fica None por padrão
        # (sem custo nenhum) — scanner/lote.py só a preenche quando o
        # detector primário configurado é o modelo treinado, caso em que
        # a busca geométrica é um complemento genuíno em vez de repetir
        # a mesma estratégia.
        self.detector_carimbo_fallback_fn = detector_carimbo_fallback_fn
        self.classificador_fn = classificador_fn or classificar_por_regras
        self.ocr_fn = ocr_fn or extrair_texto_padrao

        # Renomeação com código único de arquivamento e gravação de
        # metadados no próprio arquivo.
        self.codigo_projeto = codigo_projeto
        self.gerador_sequencial = gerador_sequencial
        self.renomeacao_habilitada = renomeacao_habilitada and gerador_sequencial is not None
        self.renomeacao_padrao = renomeacao_padrao
        self.renomeacao_digitos_sequencial = renomeacao_digitos_sequencial
        self.gravar_exif = gravar_exif

        # Organização física em pastas (ano/projeto)
        self.arquivamento_habilitado = arquivamento_habilitado
        self.arquivamento_padrao_pastas = arquivamento_padrao_pastas
        self.arquivamento_pasta_raiz = arquivamento_pasta_raiz
        self.arquivamento_copiar = arquivamento_copiar
        self.deteccao_multiorientacao = deteccao_multiorientacao
        self.nome_projeto = nome_projeto
        self.atribuicao_instituicao = atribuicao_instituicao

    def analisar_arquivo(self, caminho: Path) -> list[_AnalisePagina]:
        """FASE 1 — lê e interpreta o arquivo, SEM renomear, mover ou
        gravar nada nele.

        Separar a análise da finalização é o que permite ao lote
        propagar metadados entre as pranchas (ver
        scanner/propagacao.py) ANTES de decidir nome e pasta de
        arquivamento. Enquanto as duas coisas aconteciam juntas, o ano
        descoberto em outras pranchas do mesmo projeto chegava tarde
        demais: o arquivo já tinha sido arquivado em "Ano
        desconhecido"."""
        paginas = list(carregar_paginas(caminho))
        if not paginas:
            raise ValueError("Nenhuma página pôde ser lida do arquivo.")
        return [self._analisar_pagina(pagina) for pagina in paginas]

    def finalizar_arquivo(self, caminho: Path, analises: list[_AnalisePagina]) -> ResultadoProcessamento:
        """FASE 2 — renomeia, arquiva, grava EXIF e persiste, já com
        os metadados completos (inclusive os propagados)."""
        try:
            # A renomeação usa os metadados da primeira página (a mais
            # representativa da prancha) — arquivos multipágina viram
            # um único arquivo renomeado, com todas as páginas
            # referenciando o mesmo nome final.
            caminho_final = self._renomear_e_arquivar(caminho, analises[0].metadados, analises[0].classificacao)

            if self.gravar_exif:
                self._gravar_metadados_arquivo(caminho_final, analises[0])

            for analise in analises:
                self._persistir_pagina(caminho_final, caminho, analise)

            return ResultadoProcessamento(arquivo=caminho_final, sucesso=True)
        except Exception as exc:  # noqa: BLE001
            registrar_erro_processamento(logger, str(caminho), "pipeline", exc)
            return ResultadoProcessamento(arquivo=caminho, sucesso=False, erro=str(exc))

    def processar_arquivo(self, caminho: Path) -> ResultadoProcessamento:
        """Analisa e finaliza um arquivo isoladamente (sem propagação
        entre pranchas). Mantido para uso avulso e testes."""
        try:
            return self.finalizar_arquivo(caminho, self.analisar_arquivo(caminho))
        except Exception as exc:  # noqa: BLE001
            registrar_erro_processamento(logger, str(caminho), "pipeline", exc)
            self._salvar_registro_com_erro(caminho, str(exc))
            return ResultadoProcessamento(arquivo=caminho, sucesso=False, erro=str(exc))

    # ------------------------------------------------------------------
    # Fase 1: análise (OCR, IA, classificação) — não grava nada em disco
    # além de miniatura/carimbo, que não dependem do nome final.
    # ------------------------------------------------------------------
    def _analisar_pagina(self, pagina: ImagemCarregada) -> _AnalisePagina:
        # IMPORTANTE: só corrige orientação aqui — sem realce de
        # contraste nem redução de ruído. O detector treinado aprendeu
        # a reconhecer o carimbo em imagens "cruas" (as pranchas
        # exportadas como estão pro Roboflow, sem esse tratamento
        # extra); alimentá-lo com uma imagem visualmente diferente da
        # que ele viu no treino prejudica a detecção — foi exatamente
        # isso que causou detecção zerada mesmo em pranchas de
        # projetos que o modelo já tinha visto no treino.
        # --- Detecção do carimbo, testando as orientações possíveis ---
        imagem_processada, carimbo = self._detectar_em_qualquer_orientacao(pagina.imagem)
        texto_ocr = ""
        confianca_ocr = 0.0
        caminho_carimbo = None
        nome_temporario = f"{pagina.caminho_origem.stem}_p{pagina.indice_pagina + 1}"

        if carimbo is not None:
            recorte_carimbo = carimbo.recortar(imagem_processada)

            # O carimbo pode estar girado em relação à PÁGINA: em
            # muitos acervos ele fica numa faixa lateral estreita com
            # o texto correndo na vertical, mesmo com a prancha inteira
            # já na orientação certa. Sem corrigir a orientação do
            # RECORTE, o OCR volta vazio mesmo tendo localizado o
            # carimbo correto (observado na prática: acervo T100, onde
            # 13 de 15 pranchas tiveram o carimbo achado corretamente
            # mas nenhuma foi classificada, porque o texto estava a
            # 90°).
            recorte_carimbo = corrigir_orientacao_do_carimbo(
                recorte_carimbo, self.ocr_fn, self.idiomas_ocr)

            if self.salvar_carimbos:
                caminho_carimbo = self.exportador.salvar_carimbo(recorte_carimbo, nome_temporario)

            # Uma falha de OCR NUNCA deve derrubar o processamento do
            # arquivo inteiro: mesmo sem texto, a prancha ainda é
            # renomeada, arquivada e registrada na catalogação (e os
            # campos de identificação do projeto podem ser preenchidos
            # depois por propagação — ver scanner/propagacao.py).
            # Observado na prática: uma incompatibilidade interna do
            # PaddleOCR fez 3 arquivos de um lote falharem por
            # completo, quando só o texto deveria ter faltado.
            try:
                resultado_ocr = self._melhor_leitura_do_carimbo(recorte_carimbo)
                texto_ocr = resultado_ocr.texto
                confianca_ocr = resultado_ocr.confianca_media
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Falha no OCR de %s (%s) — prosseguindo sem texto para esta prancha.",
                    nome_temporario, exc,
                )
        else:
            logger.warning("Prosseguindo sem carimbo detectado para %s", nome_temporario)

        # --- Interpretação por IA (ou fallback por regras) ---
        metadados: MetadadosPrancha = self.interpretador_ia.interpretar(texto_ocr)

        # --- Correção inteligente via banco de conhecimento ---
        # O ano nomeia a pasta de arquivamento: um valor inválido vindo
        # do OCR ou da IA (ex.: "200K") criaria uma pasta inexistente
        # no acervo. Melhor cair em "Ano desconhecido".
        if metadados.ano and not ano_valido(metadados.ano):
            logger.info("Ano descartado por ser implausível: %r", metadados.ano)
            metadados.ano = ""

        metadados.arquiteto = self.conhecimento.sugerir_arquiteto(metadados.arquiteto) or ""
        metadados.cidade = self.conhecimento.sugerir_cidade(metadados.cidade) or ""
        metadados.projeto = self.conhecimento.sugerir_projeto(metadados.projeto) or ""
        metadados.escala = self.conhecimento.sugerir_escala(metadados.escala) or ""

        # Endereço: corrige a grafia se veio preenchido; se não veio
        # mas o projeto já foi identificado e tem um endereço
        # conhecido (de uma planilha de acervo importada, ou aprendido
        # num lote anterior), herda esse endereço — evitando depender
        # só da propagação por vizinhança dentro do MESMO lote.
        if metadados.endereco:
            metadados.endereco = self.conhecimento.sugerir_endereco(metadados.endereco) or ""
        elif metadados.projeto:
            endereco_conhecido = self.conhecimento.endereco_do_projeto(metadados.projeto)
            if endereco_conhecido:
                metadados.endereco = endereco_conhecido

        # Aprendizado automático: quando projeto e endereço aparecem
        # juntos numa prancha (lidos ou já corrigidos acima), essa
        # associação fica registrada para beneficiar outras pranchas
        # do mesmo projeto — no lote atual e em processamentos futuros.
        if metadados.projeto and metadados.endereco:
            self.conhecimento.registrar_endereco_do_projeto(metadados.projeto, metadados.endereco)

        # --- Número da folha desenhado (não escrito como texto) ---
        # Em muitos carimbos o número da folha é um algarismo grande
        # em contorno vazado, que o OCR de texto não lê. Nesse caso
        # tentamos lê-lo tratando-o como desenho (ver
        # scanner/folha_visual.py).
        #
        # O resultado vai para `numero` (o NÚMERO da folha), não para
        # `prancha` — quando a IA está disponível ela costuma usar
        # `prancha` para o TÍTULO do desenho ("PLANTA", "CORTES e
        # VISTA"), e misturar os dois fazia o título virar o
        # identificador no nome do arquivo.
        if not metadados.numero and carimbo is not None:
            numero_folha = ler_numero_folha(recorte_carimbo)
            if numero_folha:
                metadados.numero = numero_folha

        # --- Classificação automática do tipo de prancha ---
        classificacao = self.classificador_fn(texto_ocr, tipo_sugerido_ia=metadados.tipo, imagem=pagina.imagem)

        # A miniatura é gerada AQUI, enquanto a imagem ainda está em
        # memória — depois desta função a imagem é descartada, para
        # que o lote possa analisar o acervo inteiro antes de começar
        # a renomear (ver `analisar_arquivo`). O nome final do arquivo
        # ainda não é conhecido, então usamos um nome provisório que
        # será ajustado na finalização.
        caminho_miniatura = None
        if self.salvar_miniaturas:
            caminho_miniatura = self.exportador.salvar_miniatura(
                pagina.imagem, nome_temporario,
                tamanho_px=self.tamanho_miniatura,
                qualidade=self.qualidade_miniatura,
            )

        return _AnalisePagina(
            indice_pagina=pagina.indice_pagina,
            texto_ocr=texto_ocr,
            confianca_ocr=confianca_ocr,
            metadados=metadados,
            classificacao=classificacao,
            caminho_carimbo=caminho_carimbo,
            caminho_miniatura=caminho_miniatura,
        )

    @staticmethod
    def _renomear_saida_auxiliar(caminho: Optional[Path], nome_base: str) -> Optional[Path]:
        """Ajusta o nome de um arquivo auxiliar (miniatura, recorte de
        carimbo) gerado com nome provisório na fase de análise, para
        acompanhar o nome final da prancha."""
        if caminho is None or not caminho.exists():
            return caminho
        destino = caminho.with_name(f"{nome_base}{caminho.suffix}")
        if destino == caminho:
            return caminho
        try:
            caminho.replace(destino)
            return destino
        except OSError as exc:  # noqa: BLE001
            logger.debug("Não foi possível renomear arquivo auxiliar %s: %s", caminho, exc)
            return caminho

    # ------------------------------------------------------------------
    # Fase 2: renomeação do arquivo original com código único
    # ------------------------------------------------------------------
    def _melhor_leitura_do_carimbo(self, recorte):
        """Lê o carimbo tentando o recorte como está E uma versão de
        alto contraste, ficando com a melhor leitura.

        Acervos misturam digitalizações limpas com cópias desbotadas e
        traço claro sobre vegetal. Um realce forte recupera o texto das
        fracas, mas degrada as limpas — então em vez de escolher um
        tratamento fixo para todas, deixamos o resultado decidir."""
        reduzido = reduzir_para_ocr(recorte)
        leitura_direta = self.ocr_fn(reduzido, self.idiomas_ocr)
        leitura_realcada = self.ocr_fn(realcar_para_ocr(reduzido), self.idiomas_ocr)

        # "Melhor" = mais vocabulário de carimbo reconhecido; a
        # confiança média do OCR sozinha engana, porque pode vir alta
        # em pouquíssimas palavras.
        pontos_direta = _contar_acertos_aproximados(normalizar_maiusculas(leitura_direta.texto))
        pontos_realcada = _contar_acertos_aproximados(normalizar_maiusculas(leitura_realcada.texto))
        if pontos_realcada > pontos_direta:
            logger.debug("Leitura do carimbo melhorou com realce de contraste (%d vs %d palavras-chave).",
                         pontos_realcada, pontos_direta)
            return leitura_realcada
        return leitura_direta

    def _detectar_em_qualquer_orientacao(self, imagem):
        """Procura o carimbo em TODAS as orientações da prancha e fica
        com a de maior confiança.

        Motivo: um detector treinado (YOLO) NÃO é invariante a rotação
        nem a espelhamento — ele aprende a aparência do carimbo na
        orientação em que as imagens foram anotadas. Corrigir a
        orientação da prancha ANTES de detectar (pelo que maximiza o
        OCR) entregava ao modelo uma imagem girada/espelhada em
        relação ao que ele viu no treino, e a confiança despencava:
        num acervo real o modelo marcava 0.90 de mAP na validação e
        saía dando 0.06-0.24 em produção, com quase nenhuma detecção
        acima do limiar.

        Testar as orientações e escolher a de maior confiança elimina
        essa dependência: não importa como a prancha foi digitalizada
        (de lado, de cabeça para baixo ou espelhada pelo verso do
        vegetal), o carimbo é procurado em todas.

        Custa uma inferência por orientação; por isso as orientações
        espelhadas só são testadas se as diretas não derem um
        resultado convincente.

        Retorna (imagem_na_orientacao_escolhida, carimbo_ou_None)."""
        if not self.deteccao_multiorientacao:
            imagem_corrigida = corrigir_orientacao(imagem)
            carimbo = self.detector_carimbo_fn(imagem_corrigida)
            if carimbo is None:
                carimbo = self._tentar_fallback_geometrico(imagem_corrigida)
            return imagem_corrigida, carimbo

        # A BUSCA da orientação é feita numa cópia REDUZIDA. Testar as
        # 8 orientações em resolução total criava 8 cópias de uma
        # prancha de ~12000x9000 px (324 MB cada) por arquivo, vezes o
        # número de threads — o suficiente para estourar a RAM e
        # travar a máquina inteira (aconteceu de verdade). E não traz
        # ganho nenhum: o detector reduz a imagem para o tamanho de
        # inferência (imgsz) internamente de qualquer forma.
        altura, largura = imagem.shape[:2]
        escala = min(1.0, TAMANHO_BUSCA_ORIENTACAO / max(altura, largura))
        pequena = (
            cv2.resize(imagem, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
            if escala < 1.0 else imagem
        )

        melhor_transformacao = None
        melhor_confianca = -1.0
        for espelhar in (False, True):
            base = cv2.flip(pequena, 1) if espelhar else pequena
            for angulo in (0, 90, 180, 270):
                carimbo = self.detector_carimbo_fn(_rotacionar(base, angulo))
                if carimbo is not None and carimbo.confianca > melhor_confianca:
                    melhor_transformacao = (angulo, espelhar)
                    melhor_confianca = carimbo.confianca
                # Checa a CADA chamada, não só ao fim do bloco de 4
                # rotações — cada chamada ao modelo tem um custo real
                # (mais ainda em CPU), então achar uma detecção boa
                # logo na primeira rotação testada deve parar a busca
                # ali, em vez de gastar mais 3 chamadas só para
                # confirmar o que já se sabe.
                if melhor_confianca >= CONFIANCA_BOA_O_BASTANTE:
                    break
            if melhor_confianca >= CONFIANCA_BOA_O_BASTANTE:
                break

        if melhor_transformacao is None:
            # Sem carimbo em orientação nenhuma pelo detector primário
            # (comum quando o modelo em si tem desempenho ruim nesse
            # acervo — não é bug, é o modelo real ficando bem abaixo
            # da validação). Nesse caso a 2ª estratégia geométrica
            # passa a ser a ÚNICA chance de achar o carimbo — mas ela
            # também depende da orientação estar certa, e confiar só
            # no palpite de `corrigir_orientacao` (um heurístico de
            # OCR genérico, não específico de carimbo) faz a 2ª
            # estratégia buscar nas regiões erradas quando esse
            # palpite erra a rotação.
            #
            # Achado na prática: o mesmo arquivo dava 0.87 de
            # confiança na orientação certa e só 0.36 (abaixo do
            # limiar, descartado) na orientação que `corrigir_orientacao`
            # escolheu. E a pontuação puramente GEOMÉTRICA (sem OCR) não
            # serve pra escolher a orientação certa — testado na
            # prática, ela sai praticamente IDÊNTICA nas 4 rotações
            # (carimbos com grade interna são simétricos o bastante pra
            # confundir esse critério sozinho).
            #
            # Por isso, aqui testamos a 2ª estratégia COMPLETA (com
            # verificação por OCR, resolução real) nas 4 rotações
            # básicas, com saída antecipada assim que uma cruza
            # `CONFIANCA_BOA_O_BASTANTE` — não faz sentido pagar as
            # outras 3 tentativas quando a primeira já achou um
            # carimbo claramente legível. Isso custa mais tempo do que
            # antes SÓ nos arquivos onde o modelo treinado falhou por
            # completo (o pior caso já era descartado sem achar nada;
            # agora ao menos tenta de verdade).
            imagem_corrigida = None
            carimbo = None
            melhor_confianca_fallback = -1.0
            for angulo_teste in (0, 90, 180, 270):
                candidata = _rotacionar(imagem, angulo_teste)
                candidato = self._tentar_fallback_geometrico(candidata)
                if candidato is not None and candidato.confianca > melhor_confianca_fallback:
                    imagem_corrigida, carimbo = candidata, candidato
                    melhor_confianca_fallback = candidato.confianca
                if melhor_confianca_fallback >= CONFIANCA_BOA_O_BASTANTE:
                    break

            if imagem_corrigida is None:
                # nenhuma das 4 rotações achou nada: mantém o
                # comportamento anterior (orientação corrigida pelo
                # heurístico de OCR genérico, sem carimbo)
                imagem_corrigida = corrigir_orientacao(imagem)
            return imagem_corrigida, carimbo

        # Só agora a transformação vencedora é aplicada em resolução
        # total — UMA única cópia — e a detecção refeita nela, para que
        # as coordenadas do recorte saiam precisas.
        angulo, espelhar = melhor_transformacao
        imagem_final = _rotacionar(cv2.flip(imagem, 1) if espelhar else imagem, angulo)
        logger.debug("Melhor orientação para detecção: rotação=%d°, espelhado=%s (confiança %.2f)",
                     angulo, espelhar, melhor_confianca)
        carimbo = self.detector_carimbo_fn(imagem_final)
        if carimbo is None:
            carimbo = self._tentar_fallback_geometrico(imagem_final)
        return imagem_final, carimbo

    def _tentar_fallback_geometrico(self, imagem: np.ndarray) -> Optional[CarimboDetectado]:
        """Segunda estratégia de detecção, usada só quando o detector
        primário (tipicamente o modelo treinado) não encontrou nada —
        tenta a busca geométrica por regiões candidatas, verificada
        por conteúdo (palavras-chave típicas de carimbo via OCR), na
        MESMA imagem/orientação já escolhida.

        Passa a imagem em resolução ORIGINAL — `detectar_carimbo_verificado`
        (scanner/detector_carimbo.py) já cuida de reduzir só a busca
        geométrica internamente e recortar/verificar cada candidato a
        partir da resolução original, para não perder detalhe do
        texto do carimbo.

        Só é usada quando `self.detector_carimbo_fallback_fn` foi
        fornecido (ver `scanner/lote.py` — só faz sentido plugar isso
        quando o detector primário é o modelo treinado; se o primário
        já É a heurística, tentar de novo não ajudaria e só custaria
        tempo à toa). Uma tentativa por prancha, não uma por
        orientação testada — para não multiplicar o custo do OCR de
        verificação pelas até 8 chamadas da busca de orientação."""
        if self.detector_carimbo_fallback_fn is None:
            return None
        try:
            carimbo = self.detector_carimbo_fallback_fn(imagem)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Falha na 2ª estratégia de detecção de carimbo (%s).", exc)
            return None
        if carimbo is not None:
            logger.info("Carimbo encontrado na 2ª estratégia (busca geométrica), região '%s', confiança %.2f.",
                        carimbo.canto, carimbo.confianca)
        return carimbo

    def _renomear_e_arquivar(
        self, caminho_original: Path, metadados: MetadadosPrancha, classificacao: ResultadoClassificacao,
    ) -> Path:
        """Gera o arquivo final (renomeado e, se habilitado, organizado
        em pasta ano/projeto) numa ÚNICA operação a partir do arquivo
        original — o arquivo original NUNCA é renomeado/movido em
        etapas intermediárias.

        Isso é essencial: fazer "renomear o original no lugar, depois
        mover" em duas etapas separadas significa que a primeira etapa
        MUTA o arquivo original — e se o mesmo arquivo for reprocessado
        depois (ex.: com um código de projeto diferente da vez
        anterior), o nome já modificado vira a base do próximo nome,
        crescendo a cada execução (bug real observado em testes)."""
        pasta_destino = self._pasta_destino_final(metadados)

        if not self.renomeacao_habilitada:
            if pasta_destino is None:
                return caminho_original
            return arquivar_em_pasta_destino(caminho_original, pasta_destino, copiar=self.arquivamento_copiar)

        if parece_ja_renomeado(caminho_original.stem, self.codigo_projeto, self.renomeacao_digitos_sequencial):
            logger.warning(
                "Arquivo '%s' já parece ter sido renomeado pelo CAMP Vision (mesmo código de projeto) — "
                "pulando renomeação para não empilhar nomes. Se isso for um arquivo novo, use um código "
                "de projeto diferente ou renomeie-o manualmente antes de processar.",
                caminho_original.name,
            )
            if pasta_destino is None:
                return caminho_original
            return arquivar_em_pasta_destino(caminho_original, pasta_destino, copiar=self.arquivamento_copiar)

        # IMPORTANTE: nunca usar caminho_original.stem como fallback aqui.
        # O nome atual poderia já ser um nome gerado por uma execução
        # anterior (ex.: se a proteção acima não pegou por algum
        # motivo) — sem número de prancha identificado, usamos um
        # marcador fixo ("SEMNUM") em vez do nome do arquivo.
        identificador_prancha = metadados.numero or metadados.prancha or ""
        sequencial = self.gerador_sequencial.proximo(self.codigo_projeto or "PROJ")

        # Nome legível da prancha para o nome do arquivo: prioriza o
        # título lido do próprio carimbo (ex.: "PLANTA HIDRÁULICA 03");
        # só cai para tipo detectado + número (ex.: "Planta 01") quando
        # o carimbo não trouxe título legível (ver montar_nome_prancha).
        nome_prancha = montar_nome_prancha(metadados.prancha, classificacao.tipo, metadados.numero)

        novo_nome = montar_nome_arquivo(
            padrao=self.renomeacao_padrao,
            codigo_projeto=self.codigo_projeto,
            prancha=identificador_prancha,
            sequencial=sequencial,
            extensao=caminho_original.suffix,
            digitos_sequencial=self.renomeacao_digitos_sequencial,
            tipo=classificacao.tipo,
            ano=metadados.ano,
            arquiteto=metadados.arquiteto,
            endereco=metadados.endereco,
            cliente=metadados.cliente,
            fase=metadados.fase,
            projeto=metadados.projeto,
            folha=metadados.numero,
            codigo_projeto_auto=metadados.codigo_projeto_auto,
            sequencial_no_projeto=metadados.sequencial_no_projeto,
            nome_prancha=nome_prancha,
        )

        destino_final = pasta_destino if pasta_destino is not None else caminho_original.parent
        return arquivar_em_pasta_destino(
            caminho_original, destino_final, copiar=self.arquivamento_copiar, novo_nome=novo_nome,
        )

    def _pasta_destino_final(self, metadados: MetadadosPrancha) -> Optional[Path]:
        """Calcula a pasta de destino (ano/projeto, por padrão), ou
        None se a organização em pastas estiver desabilitada — nesse
        caso o arquivo final fica na mesma pasta do original.

        A pasta é organizada pelo PROJETO DESTA PRANCHA (já propagado
        e com grafias unificadas entre as pranchas do lote — ver
        scanner/propagacao.py), não pelo nome da pasta selecionada:
        um lote com vários projetos diferentes (comum quando um
        escritório digitaliza aos poucos) gera uma pasta por projeto,
        em vez de misturar tudo numa pasta só. `self.nome_projeto`
        (nome da pasta originalmente selecionada) só entra como
        retaguarda quando NENHUM projeto foi identificado nesta
        prancha."""
        if not self.arquivamento_habilitado:
            return None
        if self.arquivamento_pasta_raiz is None:
            # IMPORTANTE: a raiz precisa ser um valor fixo, calculado
            # uma única vez a partir da pasta originalmente
            # selecionada (ver scanner/lote.py) — nunca derivado do
            # caminho do arquivo em processamento, ou reprocessar um
            # arquivo já arquivado aninharia "ano/projeto" de novo
            # dentro da estrutura já criada.
            logger.error(
                "Raiz de arquivamento não definida — pulando organização em pastas "
                "(isto indica um problema de configuração; verifique scanner/lote.py)."
            )
            return None
        return montar_pasta_destino(
            raiz=self.arquivamento_pasta_raiz,
            padrao=self.arquivamento_padrao_pastas,
            ano=metadados.ano_pasta or metadados.ano,
            nome_projeto=metadados.projeto.strip() or self.nome_projeto,
            codigo_projeto_auto=metadados.codigo_projeto_auto,
        )

    # ------------------------------------------------------------------
    # Gravação de metadados no arquivo final (EXIF/IPTC/XMP)
    # ------------------------------------------------------------------
    def _gravar_metadados_arquivo(self, caminho_final: Path, analise_principal: _AnalisePagina) -> None:
        metadados = analise_principal.metadados
        gravar_metadados(
            caminho_final,
            MetadadosParaGravar(
                projeto=metadados.projeto,
                cliente=metadados.cliente,
                arquiteto=metadados.arquiteto,
                endereco=metadados.endereco,
                cidade=metadados.cidade,
                ano=metadados.ano,
                prancha=metadados.prancha,
                numero=metadados.numero,
                escala=metadados.escala,
                tipo=analise_principal.classificacao.tipo,
                fase=metadados.fase,
                observacoes=metadados.observacoes,
                codigo_gerado=caminho_final.stem,
                atribuicao_instituicao=self.atribuicao_instituicao,
            ),
        )

    # ------------------------------------------------------------------
    # Fase 3: persistência (banco + exportador)
    # ------------------------------------------------------------------
    def _persistir_pagina(
        self, caminho_final: Path, caminho_original: Path, analise: _AnalisePagina,
    ) -> None:
        nome_base = (
            caminho_final.stem if analise.indice_pagina == 0
            else f"{caminho_final.stem}_p{analise.indice_pagina + 1}"
        )
        metadados = analise.metadados
        classificacao = analise.classificacao

        # A miniatura já foi gerada na fase de análise (quando a
        # imagem ainda estava em memória); aqui só a renomeamos para
        # acompanhar o nome final do arquivo.
        caminho_miniatura = self._renomear_saida_auxiliar(analise.caminho_miniatura, nome_base)
        analise.caminho_carimbo = self._renomear_saida_auxiliar(
            analise.caminho_carimbo, f"{nome_base}_carimbo")

        registro_db = Prancha(
            arquivo=str(caminho_final),
            arquivo_original=str(caminho_original) if caminho_original != caminho_final else None,
            codigo_gerado=caminho_final.stem if self.renomeacao_habilitada else None,
            projeto=metadados.projeto,
            cliente=metadados.cliente,
            arquiteto=metadados.arquiteto,
            cidade=metadados.cidade,
            endereco=metadados.endereco,
            ano=metadados.ano,
            prancha_titulo=metadados.prancha,
            numero=metadados.numero,
            escala=metadados.escala,
            tipo=classificacao.tipo,
            fase=metadados.fase,
            observacoes=metadados.observacoes,
            confianca_ocr=analise.confianca_ocr,
            confianca_ia=metadados.confianca_ia,
            caminho_miniatura=str(caminho_miniatura) if caminho_miniatura else None,
            caminho_carimbo=str(analise.caminho_carimbo) if analise.caminho_carimbo else None,
        )
        self.pranchas_repo.salvar(registro_db)

        self.exportador.adicionar_registro(RegistroExportacao(
            arquivo=caminho_final.name,
            arquivo_original=caminho_original.name if caminho_original != caminho_final else "",
            codigo_projeto_auto=metadados.codigo_projeto_auto,
            projeto=metadados.projeto,
            cliente=metadados.cliente,
            arquiteto=metadados.arquiteto,
            cidade=metadados.cidade,
            endereco=metadados.endereco,
            ano=metadados.ano,
            prancha=metadados.prancha,
            numero=metadados.numero,
            escala=metadados.escala,
            tipo=classificacao.tipo,
            fase=metadados.fase,
            observacoes=metadados.observacoes,
            confianca_ocr=analise.confianca_ocr,
            confianca_ia=metadados.confianca_ia,
        ))

        logger.info("Processado: %s -> tipo=%s, arquiteto=%s", nome_base, classificacao.tipo, metadados.arquiteto)

    def _salvar_registro_com_erro(self, caminho: Path, mensagem_erro: str) -> None:
        registro_db = Prancha(arquivo=str(caminho), erro=mensagem_erro)
        self.pranchas_repo.salvar(registro_db)
        self.exportador.adicionar_registro(RegistroExportacao(arquivo=caminho.name, observacoes=f"ERRO: {mensagem_erro}"))
