"""
scanner/detector_carimbo.py
============================
Localização automática do carimbo (selo de identificação) em pranchas
arquitetônicas — sem assumir posição fixa, já que ela pode variar
livremente de folha para folha dentro do mesmo acervo.

Estratégia (duas camadas):

  1. GEOMETRIA: a imagem é dividida em regiões candidatas (os quatro
     cantos e as quatro faixas completas de borda — comum em
     convenções brasileiras onde o carimbo ocupa uma faixa vertical
     ou horizontal inteira). Em cada região, localizamos o retângulo
     com borda fechada mais provável (aproximação poligonal de ~4
     vértices, alta solidez área/bounding-box, tamanho relativo).

  2. CONTEÚDO: quando há mais de uma região candidata plausível (ou
     seja, quando não se sabe de antemão onde o carimbo está), o
     geometricamente "melhor de cada região" é submetido a uma
     verificação por OCR — o candidato cujo texto reconhecido mais se
     parece com um carimbo real (contém palavras como "ESCALA",
     "ARQUITETO", "PROJETO" etc.) vence, mesmo que sua geometria não
     fosse a nota mais alta. Isso é necessário porque, em desenhos
     técnicos cheios de linhas, a geometria sozinha erra com
     frequência — o conteúdo é o sinal definitivo de que "isto é
     mesmo um carimbo", já que rodamos OCR de qualquer forma no
     restante do pipeline.

  3. Caso nenhuma região produza um candidato com confiança mínima
     (geométrica ou de conteúdo), o detector retorna None e o
     chamador deve registrar o erro e seguir o processamento (a
     ausência de carimbo não interrompe o lote).

Uma região específica pode ser fixada via `criar_detector(...,
regiao_fixa=...)` quando se sabe, de antemão, que o carimbo está
sempre no mesmo lugar naquele acervo — nesse caso a verificação por
conteúdo é pulada (só há um candidato, não há o que comparar).
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from scanner.leitor_imagem import reduzir_para_ocr
from utils.texto import normalizar_maiusculas as _normalizar_para_comparacao

logger = logging.getLogger("campvision.scanner.carimbo")

PROPORCAO_CANTO = 0.32     # fração da largura/altura usada como região de "canto"
PROPORCAO_FAIXA = 0.16     # fração usada como espessura de "faixa" de borda inteira
AREA_MINIMA_RELATIVA = 0.01
# A área é relativa à REGIÃO DE BUSCA (já um canto/faixa pequeno da
# prancha, não a prancha inteira) — por isso o carimbo pode
# legitimamente ocupar quase toda essa região.
AREA_MAXIMA_RELATIVA = 0.97
PONTUACAO_MINIMA = 0.30

# Palavras e abreviações tipicamente encontradas em carimbos de
# arquitetura brasileiros — usadas para verificar, por conteúdo, qual
# candidato geométrico é de fato o carimbo (ver `_pontuar_conteudo`).
# Inclui abreviações comuns (ENG., CONSTR., EMPR.) além das palavras
# por extenso, já que testes com acervos reais mostraram carimbos que
# usam quase só abreviações e nenhuma das palavras "óbvias".
PALAVRAS_CHAVE_CARIMBO = (
    "ESCALA", "PROJETO", "ARQUITETO", "ARQUITETURA", "CLIENTE", "DATA",
    "PRANCHA", "PROPRIETARIO", "ENDERECO", "CIDADE", "FASE", "DESENHO",
    "DESENHOU", "RESPONSAVEL", "CREA", "CAU", "REV", "REVISAO", "FOLHA",
    "OBRA", "LOCAL", "MUNICIPIO", "ESTADO", "APROVADO", "VERIFICADO",
    "RUA", "AV", "AVENIDA", "ALAMEDA", "TELEFONE", "TEL", "FONE",
    "LTDA", "S.A", "S/A", "EIRELI", "ENG", "ARQ", "CONSTR", "EMPR",
    "IMOB", "URBANISMO", "ENGENHARIA", "N.", "Nº", "EX.",
)

_PADRAO_TELEFONE = re.compile(r"\b\d{3,5}[\s.\-]?\d{4}\b")

REGIOES_VALIDAS = (
    "automatico",
    "superior_esquerdo", "superior_direito", "inferior_esquerdo", "inferior_direito",
    "faixa_direita", "faixa_inferior", "faixa_esquerda", "faixa_superior",
)


@dataclass
class CarimboDetectado:
    x: int
    y: int
    largura: int
    altura: int
    confianca: float
    canto: str

    def recortar(self, imagem: np.ndarray) -> np.ndarray:
        return imagem[self.y: self.y + self.altura, self.x: self.x + self.largura].copy()


def _regioes_candidatas(altura_img: int, largura_img: int, regiao_fixa: str = "automatico") -> dict[str, tuple[int, int, int, int]]:
    rh_canto = int(altura_img * PROPORCAO_CANTO)
    rw_canto = int(largura_img * PROPORCAO_CANTO)
    fh = int(altura_img * PROPORCAO_FAIXA)
    fw = int(largura_img * PROPORCAO_FAIXA)

    todas = {
        "superior_esquerdo": (0, 0, rw_canto, rh_canto),
        "superior_direito": (largura_img - rw_canto, 0, largura_img, rh_canto),
        "inferior_esquerdo": (0, altura_img - rh_canto, rw_canto, altura_img),
        "inferior_direito": (largura_img - rw_canto, altura_img - rh_canto, largura_img, altura_img),
        "faixa_direita": (largura_img - fw, 0, largura_img, altura_img),
        "faixa_inferior": (0, altura_img - fh, largura_img, altura_img),
        "faixa_esquerda": (0, 0, fw, altura_img),
        "faixa_superior": (0, 0, largura_img, fh),
    }

    if regiao_fixa == "automatico":
        return todas
    if regiao_fixa in todas:
        return {regiao_fixa: todas[regiao_fixa]}

    logger.warning("Região de carimbo '%s' desconhecida — usando busca automática.", regiao_fixa)
    return todas


def _pontuar_contorno(contorno: np.ndarray, area_regiao: float, regiao_recortada: np.ndarray) -> float:
    x, y, w, h = cv2.boundingRect(contorno)
    area_bbox = w * h
    if area_bbox == 0:
        return 0.0

    area_relativa = area_bbox / area_regiao
    if not (AREA_MINIMA_RELATIVA <= area_relativa <= AREA_MAXIMA_RELATIVA):
        return 0.0

    area_contorno = cv2.contourArea(contorno)
    solidez = area_contorno / area_bbox  # 1.0 = preenche perfeitamente o retângulo delimitador

    perimetro = cv2.arcLength(contorno, True)
    aproximado = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
    eh_quadrilatero = len(aproximado) in (4, 5, 6)  # tolera pequenas imperfeições na borda

    pontuacao_retangularidade = solidez if eh_quadrilatero else solidez * 0.4
    pontuacao_tamanho = min(area_relativa / AREA_MAXIMA_RELATIVA, 1.0)

    proporcao = w / h if h else 0
    pontuacao_proporcao = 1.0 if 0.3 <= proporcao <= 8.0 else 0.5

    recorte = regiao_recortada[y:y + h, x:x + w]
    densidade_bordas = 0.0
    if recorte.size > 0:
        bordas_internas = cv2.Canny(recorte, 50, 150)
        densidade_bordas = float(np.count_nonzero(bordas_internas)) / bordas_internas.size
    pontuacao_densidade = min(densidade_bordas * 6, 1.0)

    return (
        0.40 * pontuacao_retangularidade
        + 0.30 * pontuacao_tamanho
        + 0.15 * pontuacao_proporcao
        + 0.15 * pontuacao_densidade
    )


def _melhor_candidato_por_regiao(imagem: np.ndarray, regiao_fixa: str) -> dict[str, CarimboDetectado]:
    """Para cada região candidata, retorna o contorno geometricamente
    mais provável — sem ainda decidir qual região é a vencedora.

    Muitos carimbos de título são divididos em DUAS caixas lado a
    lado — ex.: bloco com nome/endereço do escritório à esquerda e
    tabela institucional (cliente, projeto, escala, revisões) à
    direita, cada uma com sua própria borda. Escolher só o contorno
    de maior pontuação cortaria fora a outra caixa (achado com um
    carimbo real: só a caixa do escritório era capturada, perdendo
    justamente a tabela com cliente/projeto/escala). Por isso, depois
    de achar o melhor candidato, outras caixas na MESMA faixa vertical
    e horizontalmente PRÓXIMAS a ele são fundidas numa única região."""
    altura_img, largura_img = imagem.shape[:2]
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    candidatos: dict[str, CarimboDetectado] = {}

    for nome_regiao, (x0, y0, x1, y1) in _regioes_candidatas(altura_img, largura_img, regiao_fixa).items():
        regiao = cinza[y0:y1, x0:x1]
        if regiao.size == 0:
            continue

        binarizada = cv2.adaptiveThreshold(
            regiao, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        fechada = cv2.morphologyEx(binarizada, cv2.MORPH_CLOSE, kernel)

        contornos, _ = cv2.findContours(fechada, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        area_regiao = regiao.shape[0] * regiao.shape[1]

        caixas_validas: list[tuple[float, tuple[int, int, int, int]]] = []
        for contorno in contornos:
            pontuacao = _pontuar_contorno(contorno, area_regiao, regiao)
            if pontuacao < PONTUACAO_MINIMA:
                continue
            caixas_validas.append((pontuacao, cv2.boundingRect(contorno)))

        if not caixas_validas:
            continue

        caixas_validas.sort(key=lambda c: c[0], reverse=True)
        melhor_pontuacao, (cx0, cy0, cw, ch) = caixas_validas[0]
        cx1, cy1 = cx0 + cw, cy0 + ch

        for pontuacao, (x, y, w, h) in caixas_validas[1:]:
            sobreposicao_vertical = min(cy1, y + h) - max(cy0, y)
            altura_menor = min(cy1 - cy0, h)
            if altura_menor <= 0 or sobreposicao_vertical / altura_menor < 0.6:
                continue  # não está na mesma faixa vertical do carimbo
            gap_horizontal = max(0, max(x, cx0) - min(x + w, cx1))
            if gap_horizontal > 0.15 * (cx1 - cx0):
                continue  # longe demais pra ser a mesma caixa dividida
            cx0, cy0 = min(cx0, x), min(cy0, y)
            cx1, cy1 = max(cx1, x + w), max(cy1, y + h)

        candidatos[nome_regiao] = CarimboDetectado(
            x=x0 + cx0, y=y0 + cy0, largura=cx1 - cx0, altura=cy1 - cy0,
            confianca=melhor_pontuacao, canto=nome_regiao,
        )

    return candidatos


LIMIAR_SIMILARIDADE_PALAVRA_CHAVE = 0.72  # 0-1 (difflib) — tolera erros de OCR sem aceitar qualquer coisa


def _palavra_corresponde_a_chave(palavra: str, chave: str) -> bool:
    """Compara uma palavra do OCR com uma palavra-chave, tolerando
    erros de reconhecimento — mas exigindo que os tamanhos sejam
    parecidos, para não confundir uma palavra curta e comum (ex.:
    "SALA", nome de cômodo) com uma chave mais longa que a contém
    como substring (ex.: "ESCALA") só por coincidência de letras."""
    if len(chave) < 3:
        return palavra == chave
    maior, menor = max(len(palavra), len(chave)), min(len(palavra), len(chave))
    if menor / maior < 0.75:
        return False
    return difflib.SequenceMatcher(None, palavra, chave).ratio() >= LIMIAR_SIMILARIDADE_PALAVRA_CHAVE


def _contar_acertos_aproximados(texto_normalizado: str) -> int:
    """Conta quantas palavras do texto reconhecido 'parecem' com
    alguma palavra-chave de carimbo, usando similaridade textual
    (difflib) em vez de correspondência exata.

    Isso é necessário porque folhas degradadas produzem OCR cheio de
    pequenos erros — "TELEF0NE", "ARQUITET0", "ESGALA" — e exigir
    correspondência exata faz o sistema simplesmente não reconhecer
    o carimbo nesses casos. A mesma técnica já é usada em
    `database/repository.py` para corrigir nomes de arquitetos com
    erro de OCR; aqui aplicamos o mesmo princípio a palavras isoladas
    do texto do carimbo."""
    acertos = 0
    for palavra in texto_normalizado.split():
        if len(palavra) < 3:
            continue
        for chave in PALAVRAS_CHAVE_CARIMBO:
            if _palavra_corresponde_a_chave(palavra, chave):
                acertos += 1
                break
    return acertos


def _pontuar_conteudo(texto: str) -> float:
    """Pontua o quanto um texto reconhecido por OCR 'parece' um
    carimbo real de arquitetura.

    Combina três sinais:
    - presença de palavras-chave/abreviações típicas de carimbo,
      comparadas de forma APROXIMADA (tolerante a erros de OCR —
      ver `_contar_acertos_aproximados`) — folhas degradadas raramente
      produzem a grafia exata de qualquer palavra;
    - presença de um padrão de telefone (ex. "853-4530") — carimbos
      de arquitetura praticamente sempre trazem um telefone de
      contato, e cortes/plantas/detalhes praticamente nunca têm um
      padrão assim;
    - densidade de texto reconhecido, como sinal bem mais fraco (só
      desempata entre candidatos igualmente "sem cara de carimbo" —
      testes reais mostraram que dar peso alto à densidade faz o
      sistema escolher por engano folhas de detalhes ou legendas só
      por terem bastante texto, mesmo sem nenhuma palavra de carimbo)."""
    texto_normalizado = _normalizar_para_comparacao(texto)
    acertos = _contar_acertos_aproximados(texto_normalizado)
    pontuacao_palavras_chave = min(acertos / 3, 1.0)

    pontuacao_telefone = 1.0 if _PADRAO_TELEFONE.search(texto) else 0.0

    quantidade_palavras = len(texto.split())
    pontuacao_densidade = min(quantidade_palavras / 40, 1.0)

    return (
        0.55 * pontuacao_palavras_chave
        + 0.30 * pontuacao_telefone
        + 0.15 * pontuacao_densidade
    )


# Lado máximo da cópia da PÁGINA INTEIRA usada só para a busca por
# contorno (achar ONDE estão as regiões candidatas) — essa etapa
# (adaptiveThreshold/morphologyEx/findContours/Canny em até 8 regiões,
# algumas cobrindo quase a largura/altura inteira da prancha) é cara
# demais em resolução original (pranchas de até ~12000x9000px). A
# verificação por OCR de cada candidato, por outro lado, recorta da
# imagem ORIGINAL (ver `detectar_carimbo_verificado`) — reduzir a
# página inteira ANTES de recortar o candidato deixava o texto do
# carimbo pequeno/borrado demais para o OCR reconhecer as
# palavras-chave, mesmo em carimbos de verdade (confirmado: a taxa de
# detecção caiu quase a zero quando essa redução acontecia antes do
# recorte, em vez de depois).
TAMANHO_MAX_BUSCA_CONTORNO = 2200


def detectar_carimbo(imagem: np.ndarray, regiao_fixa: str = "automatico") -> Optional[CarimboDetectado]:
    """Versão puramente geométrica: procura o carimbo nas regiões
    candidatas e retorna o melhor candidato entre todas elas, sem
    verificação por conteúdo (usada como base por
    `detectar_carimbo_verificado`, e diretamente quando não há motor
    de OCR disponível para a verificação)."""
    candidatos = _melhor_candidato_por_regiao(imagem, regiao_fixa)

    if not candidatos:
        logger.warning("Carimbo não localizado com confiança suficiente (região='%s')", regiao_fixa)
        return None

    melhor = max(candidatos.values(), key=lambda c: c.confianca)
    logger.debug("Carimbo detectado em '%s' com confiança geométrica %.2f", melhor.canto, melhor.confianca)
    return melhor


TAMANHO_MAX_VERIFICACAO = 1400  # px no maior lado — suficiente para achar palavras-chave, muito mais rápido


def _reduzir_para_verificacao(recorte: np.ndarray) -> np.ndarray:
    """Reduz um recorte candidato para um tamanho manejável antes de
    rodar OCR de verificação por conteúdo. Pranchas de arquitetura
    digitalizadas costumam ser enormes (mais de 10000px de lado), e
    algumas regiões candidatas (as faixas de borda inteiras) chegam a
    cobrir quase a altura/largura toda da prancha — rodar OCR nelas em
    resolução original, até 8 vezes por arquivo, é o que deixava o
    processamento muito lento. Para só detectar palavras-chave, uma
    versão bem menor é suficiente."""
    return reduzir_para_ocr(recorte, TAMANHO_MAX_VERIFICACAO)


# Pontuação final mínima (0-1, mesma escala de `pontuacao_final` em
# `detectar_carimbo_verificado`) para aceitar um candidato verificado
# por conteúdo. Sem isso, a função sempre devolvia o candidato de
# MAIOR pontuação entre as regiões, mesmo quando nenhuma delas de
# fato parecia um carimbo — um bloco de legenda ou detalhe construtivo
# (texto técnico denso, sem nenhuma palavra-chave real de carimbo)
# passava só pela densidade de texto, com pontuação típica de
# 0.20-0.31 — bem abaixo do que um carimbo de verdade costuma
# pontuar (0.40+, geralmente com telefone e várias palavras-chave
# batendo). Achado com casos reais (recortes confirmados como legenda
# de fachada / corte / detalhe, não carimbo, aceitos com confiança
# 0.22 a 0.31).
LIMIAR_MINIMO_CONTEUDO_VERIFICADO = 0.40


def detectar_carimbo_verificado(
    imagem: np.ndarray,
    ocr_fn: Callable[[np.ndarray, list], object],
    idiomas_ocr: Optional[list[str]] = None,
    regiao_fixa: str = "automatico",
    limiar_minimo: float = LIMIAR_MINIMO_CONTEUDO_VERIFICADO,
) -> Optional[CarimboDetectado]:
    """Como `detectar_carimbo`, mas usa OCR para verificar se a região
    candidata de fato contém texto de carimbo (palavras-chave
    típicas), em vez de confiar apenas na geometria — essencial
    quando a posição do carimbo varia livremente entre as pranchas do
    acervo, e mesmo quando só existe UM candidato geométrico (a
    maioria das regiões buscadas não passa nem do filtro geométrico
    básico, então "só um candidato" é o caso mais comum, não uma
    exceção rara — pular a verificação nesse caso deixava passar
    candidatos sem nenhuma palavra-chave de carimbo real).

    A busca por contorno (achar ONDE estão as regiões candidatas) roda
    numa cópia REDUZIDA da página inteira (ver
    `TAMANHO_MAX_BUSCA_CONTORNO`) — é a etapa cara. Mas o RECORTE de
    cada candidato para a verificação por OCR (e o resultado final
    devolvido) vem da imagem ORIGINAL, não da cópia reduzida — reduzir
    a página inteira antes de recortar deixaria o texto do candidato
    pequeno/borrado demais para o OCR reconhecer, mesmo num carimbo de
    verdade.

    Só aceita o candidato de maior pontuação se ela cruzar
    `limiar_minimo` — caso contrário, nenhuma das regiões realmente
    parece um carimbo (ver `LIMIAR_MINIMO_CONTEUDO_VERIFICADO`), e é
    melhor devolver None do que arquivar um recorte de legenda ou
    detalhe construtivo como se fosse o carimbo."""
    altura_original, largura_original = imagem.shape[:2]
    escala_busca = min(1.0, TAMANHO_MAX_BUSCA_CONTORNO / max(altura_original, largura_original))
    imagem_busca = (
        cv2.resize(imagem, None, fx=escala_busca, fy=escala_busca, interpolation=cv2.INTER_AREA)
        if escala_busca < 1.0 else imagem
    )

    candidatos = _melhor_candidato_por_regiao(imagem_busca, regiao_fixa)

    if not candidatos:
        logger.warning("Carimbo não localizado com confiança suficiente (região='%s')", regiao_fixa)
        return None

    # Escala as coordenadas dos candidatos de volta pra resolução
    # ORIGINAL antes de recortar — a partir daqui, tudo (recorte pra
    # verificação, resultado final) usa a imagem original.
    if escala_busca < 1.0:
        fator = 1.0 / escala_busca
        candidatos = {
            nome: CarimboDetectado(
                x=int(c.x * fator), y=int(c.y * fator),
                largura=int(c.largura * fator), altura=int(c.altura * fator),
                confianca=c.confianca, canto=c.canto,
            )
            for nome, c in candidatos.items()
        }

    # IMPORTANTE: não existe mais atalho para quando há só UM candidato
    # geométrico. Antes, esse caso devolvia o candidato direto, sem
    # verificação por OCR nem checagem do limiar mínimo — na prática,
    # a maioria das 8 regiões buscadas não passa nem do filtro
    # geométrico básico, então "só um candidato" era o caso mais
    # comum, não uma exceção rara. Isso deixava passar candidatos sem
    # nenhuma palavra-chave de carimbo real, sem log nenhum (achado
    # investigando um arquivo real onde só 1 de 4 tentativas de
    # orientação aparecia no log — as outras 3 tinham encontrado
    # candidato único e retornado em silêncio, sem verificação).
    # Agora todo candidato passa pela mesma verificação de conteúdo,
    # mesmo sendo o único da rodada.
    melhor: Optional[CarimboDetectado] = None
    melhor_pontuacao_final = -1.0

    for nome_regiao, candidato in candidatos.items():
        recorte = candidato.recortar(imagem)
        recorte_para_verificacao = _reduzir_para_verificacao(recorte)
        try:
            resultado_ocr = ocr_fn(recorte_para_verificacao, idiomas_ocr or ["pt"])
            pontuacao_conteudo = _pontuar_conteudo(resultado_ocr.texto)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Falha ao verificar conteúdo do candidato '%s': %s", nome_regiao, exc)
            pontuacao_conteudo = 0.0

        pontuacao_final = 0.35 * candidato.confianca + 0.65 * pontuacao_conteudo
        logger.debug(
            "Candidato '%s': geometria=%.2f, conteúdo=%.2f, final=%.2f",
            nome_regiao, candidato.confianca, pontuacao_conteudo, pontuacao_final,
        )

        if pontuacao_final > melhor_pontuacao_final:
            melhor_pontuacao_final = pontuacao_final
            candidato.confianca = pontuacao_final  # reflete a pontuação combinada no resultado final
            melhor = candidato

    if melhor is None:
        return None

    if melhor_pontuacao_final < limiar_minimo:
        logger.info(
            "Melhor candidato por conteúdo (região '%s', confiança %.2f) abaixo do limiar mínimo (%.2f) — "
            "provavelmente não é um carimbo (legenda/detalhe com texto denso mas sem palavra-chave real).",
            melhor.canto, melhor_pontuacao_final, limiar_minimo,
        )
        return None

    logger.info("Carimbo escolhido por conteúdo: região '%s', confiança final %.2f",
                 melhor.canto, melhor.confianca)
    return melhor


def criar_detector(
    modo: str = "heuristico",
    caminho_modelo: Optional[str] = None,
    confianca_minima: float = 0.5,
    regiao_fixa: str = "automatico",
    ocr_fn: Optional[Callable[[np.ndarray, list], object]] = None,
    idiomas_ocr: Optional[list[str]] = None,
    tamanho_imagem_ml: int = 0,
) -> Callable[[np.ndarray], Optional["CarimboDetectado"]]:
    """Fábrica de estratégia: retorna a função de detecção de carimbo
    a ser usada pelo pipeline, conforme a configuração do usuário.

    - "heuristico": usa a busca geométrica por regiões. Se `ocr_fn`
      for fornecido (o pipeline sempre fornece, usando o mesmo motor
      de OCR configurado), a escolha final entre regiões candidatas é
      verificada por conteúdo (`detectar_carimbo_verificado`) —
      essencial quando a posição do carimbo varia entre pranchas.
      `regiao_fixa` restringe a busca a uma única região quando a
      posição é conhecida e constante, pulando a verificação (só há
      um candidato).
    - "modelo_treinado": carrega um modelo YOLO treinado
      (ver `ml/treinar_carimbo.py`) e usa sua inferência.
      `tamanho_imagem_ml=0` (padrão) faz o tamanho de imagem ser
      detectado automaticamente a partir do próprio arquivo do
      modelo — o usuário não precisa saber nem informar esse valor.
      Um valor diferente de 0 força um tamanho específico (só
      necessário em casos excepcionais). Se o carregamento do modelo
      falhar por qualquer motivo, o sistema registra o erro e cai
      automaticamente para a heurística.
    """
    if modo == "modelo_treinado":
        if not caminho_modelo:
            logger.warning("Modo 'modelo_treinado' selecionado sem caminho de modelo — usando heurística.")
        else:
            try:
                from ml.detector_carimbo_ml import DetectorCarimboML
                modelo = DetectorCarimboML(
                    caminho_modelo,
                    confianca_minima=confianca_minima,
                    tamanho_imagem=tamanho_imagem_ml if tamanho_imagem_ml > 0 else None,
                )
                logger.info("Detector de carimbo por modelo treinado carregado: %s", caminho_modelo)
                return modelo.detectar
            except Exception as exc:  # noqa: BLE001
                logger.error("Falha ao carregar modelo treinado de carimbo (%s). Usando heurística.", exc)

    if ocr_fn is not None:
        def _detector_verificado(imagem: np.ndarray) -> Optional[CarimboDetectado]:
            return detectar_carimbo_verificado(imagem, ocr_fn, idiomas_ocr, regiao_fixa=regiao_fixa)
        return _detector_verificado

    def _detector_heuristico(imagem: np.ndarray) -> Optional[CarimboDetectado]:
        return detectar_carimbo(imagem, regiao_fixa=regiao_fixa)

    return _detector_heuristico


def corrigir_orientacao_do_carimbo(recorte: np.ndarray, ocr_fn, idiomas: list) -> np.ndarray:
    """Endireita o RECORTE do carimbo escolhendo, entre as oito
    orientações possíveis, aquela cujo texto mais parece um carimbo.

    Difere de `corrigir_orientacao` (usada na página inteira), que
    decide pela confiança bruta do OCR. Num recorte de carimbo essa
    confiança quase empata entre orientações: são campos curtos em
    letra técnica, e o Tesseract devolve pontuações parecidas para
    texto de cabeça para baixo ou espelhado. O resultado era recorte
    salvo deitado — e, pior, o OCR daquela prancha saía ilegível.

    Aqui a orientação vencedora é a que reconhece mais VOCABULÁRIO de
    carimbo (ESCALA, DATA, ARQUITETO, RUA, PROJETO...). Esse critério
    é direto ao ponto: a orientação certa é justamente aquela em que
    as palavras do carimbo aparecem. A confiança do OCR entra só como
    desempate."""
    from scanner.leitor_imagem import _gerar_transformacoes, _rotacionar, reduzir_para_ocr

    reduzido = reduzir_para_ocr(recorte, 1400)
    melhor_chave, melhor_pontuacao = (0, False), (-1, -1.0)

    for (angulo, espelhado), candidata in _gerar_transformacoes(reduzido).items():
        try:
            resultado = ocr_fn(candidata, idiomas)
        except Exception:  # noqa: BLE001
            continue
        pontuacao = (
            _contar_acertos_aproximados(_normalizar_para_comparacao(resultado.texto)),
            resultado.confianca_media,
        )
        if pontuacao > melhor_pontuacao:
            melhor_chave, melhor_pontuacao = (angulo, espelhado), pontuacao

    angulo, espelhado = melhor_chave
    if not angulo and not espelhado:
        return recorte

    logger.info("Carimbo endireitado (rotação=%d°, espelhado=%s; %d palavra(s) de carimbo reconhecida(s)).",
                angulo, espelhado, melhor_pontuacao[0])
    base = cv2.flip(recorte, 1) if espelhado else recorte
    return _rotacionar(base, angulo)
