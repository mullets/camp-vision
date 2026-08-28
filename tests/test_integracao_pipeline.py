"""
Teste de integração: exercita o pipeline de detecção de carimbo e
classificação sobre uma imagem sintética gerada em memória, sem
depender de arquivos TIFF reais nem de chamadas externas de OCR/IA.
"""

import numpy as np
import cv2

from scanner.detector_carimbo import detectar_carimbo
from classificacao.classificador import classificar


def _criar_imagem_sintetica_com_carimbo() -> np.ndarray:
    """Gera uma imagem branca simulando uma prancha, com um retângulo
    com textura (simulando um carimbo com texto) no canto inferior
    direito."""
    imagem = np.full((1200, 1600, 3), 255, dtype=np.uint8)

    x0, y0, x1, y1 = 1300, 1000, 1580, 1180
    cv2.rectangle(imagem, (x0, y0), (x1, y1), (0, 0, 0), 2)
    # Simula linhas de texto dentro do carimbo (várias linhas horizontais)
    for y in range(y0 + 20, y1 - 10, 15):
        cv2.line(imagem, (x0 + 10, y), (x1 - 10, y), (0, 0, 0), 1)

    return imagem


def test_deteccao_carimbo_em_imagem_sintetica():
    imagem = _criar_imagem_sintetica_com_carimbo()
    carimbo = detectar_carimbo(imagem)
    # Em uma imagem sintética simples, a detecção pode ou não atingir
    # o limiar de confiança — o importante é que não lança exceção e,
    # quando encontra, localiza a região no canto correto.
    if carimbo is not None:
        assert carimbo.canto == "inferior_direito"


def _imagem_com_carimbo_dividido_em_duas_caixas() -> np.ndarray:
    """Reproduz um padrão real de carimbo dividido em duas caixas
    lado a lado (bloco de firma/endereço + tabela institucional com
    cliente/projeto/escala), coladas ou quase coladas — comum em
    pranchas de acervo, achado com um TIFF real."""
    imagem = np.full((2000, 3000, 3), 255, dtype=np.uint8)
    cv2.rectangle(imagem, (2060, 1450), (2400, 1900), (0, 0, 0), 3)
    cv2.putText(imagem, "FIRMA", (2090, 1600), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.rectangle(imagem, (2405, 1450), (2950, 1900), (0, 0, 0), 3)
    cv2.putText(imagem, "ELETROPAULO", (2430, 1550), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(imagem, "ESCALA 1 100", (2430, 1650), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return imagem


def test_caixas_adjacentes_sao_fundidas_em_uma_regiao_so():
    """Achado com um TIFF real (prancha unida pelo CAMP União): o
    carimbo tinha duas caixas lado a lado, e a busca por contorno
    capturava só a de maior pontuação, cortando a outra fora — nesse
    caso específico, cortava justamente a tabela com cliente, projeto
    e escala, mantendo só o bloco com nome/endereço do escritório.
    Agora caixas na mesma faixa vertical e próximas horizontalmente
    são fundidas numa região só."""
    from scanner.detector_carimbo import _melhor_candidato_por_regiao

    imagem = _imagem_com_carimbo_dividido_em_duas_caixas()
    candidatos = _melhor_candidato_por_regiao(imagem, "inferior_direito")

    carimbo = candidatos["inferior_direito"]
    # a região final precisa cobrir as DUAS caixas (de x=2060 a x=2950),
    # não só uma delas
    assert carimbo.x <= 2070
    assert carimbo.x + carimbo.largura >= 2940


def test_caixas_distantes_nao_sao_fundidas():
    """Confirma que a fusão não é agressiva demais: duas caixas na
    mesma faixa vertical mas bem separadas uma da outra continuam
    sendo tratadas como candidatas distintas, e só a de maior
    pontuação é escolhida — sem "inventar" uma região gigante que
    engoliria o resto da prancha."""
    from scanner.detector_carimbo import _melhor_candidato_por_regiao

    imagem = np.full((2000, 3000, 3), 255, dtype=np.uint8)
    cv2.rectangle(imagem, (2060, 1450), (2200, 1900), (0, 0, 0), 3)
    cv2.putText(imagem, "A", (2090, 1650), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.rectangle(imagem, (2850, 1450), (2990, 1900), (0, 0, 0), 3)
    cv2.putText(imagem, "B", (2880, 1650), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    candidatos = _melhor_candidato_por_regiao(imagem, "inferior_direito")

    carimbo = candidatos["inferior_direito"]
    assert carimbo.largura < 300  # pegou só UMA das duas caixas, não as duas


def test_candidato_unico_sem_palavra_chave_e_rejeitado():
    """Achado investigando um arquivo real: quando a busca geométrica
    encontra candidato em SÓ UMA região (o caso mais comum, já que a
    maioria das regiões buscadas não passa nem do filtro geométrico
    básico), a função pulava a verificação por OCR e o limiar mínimo
    inteiramente — devolvia o candidato direto, sem checar se ele
    tinha cara de carimbo de verdade, e sem logar nada. Isso deixava
    passar (ou silenciosamente descartar, sem registro) candidatos que
    nunca tinham sido verificados. Agora candidato único passa pela
    MESMA verificação de conteúdo que candidatos múltiplos."""
    from scanner.detector_carimbo import detectar_carimbo_verificado

    imagem = np.full((1200, 1600, 3), 255, dtype=np.uint8)
    cv2.rectangle(imagem, (1100, 800), (1580, 1150), (0, 0, 0), 2)
    for y in range(820, 1140, 12):
        cv2.line(imagem, (1110, y), (1570, y), (0, 0, 0), 1)

    class _ResultadoOCRFalso:
        def __init__(self, texto):
            self.texto = texto

    def _ocr_sem_palavra_chave(imagem, idiomas):
        return _ResultadoOCRFalso("BLOCOS DE CONCRETO JUNTAS A PRUMO APARENTES " * 3)

    resultado = detectar_carimbo_verificado(imagem, _ocr_sem_palavra_chave)
    assert resultado is None


def test_candidato_unico_com_palavra_chave_real_continua_aceito():
    from scanner.detector_carimbo import detectar_carimbo_verificado

    imagem = np.full((1200, 1600, 3), 255, dtype=np.uint8)
    cv2.rectangle(imagem, (1100, 800), (1580, 1150), (0, 0, 0), 2)
    for y in range(820, 1140, 12):
        cv2.line(imagem, (1110, y), (1570, y), (0, 0, 0), 1)

    class _ResultadoOCRFalso:
        def __init__(self, texto):
            self.texto = texto

    def _ocr_com_palavras_chave(imagem, idiomas):
        return _ResultadoOCRFalso("ARQUITETO ESCALA 1 100 PROJETO TELEFONE 853 4530 DATA REVISAO")

    resultado = detectar_carimbo_verificado(imagem, _ocr_com_palavras_chave)
    assert resultado is not None
    assert resultado.canto == "inferior_direito"


def test_verificacao_por_conteudo_rejeita_regiao_sem_cara_de_carimbo():
    """Reproduz um problema real: pranchas de acervo com blocos de
    texto técnico denso (legenda de fachada, especificação de
    material) mas SEM nenhuma palavra-chave real de carimbo estavam
    sendo aceitas como carimbo por pura densidade de texto, com
    confiança 0.22-0.31 — confirmado com recortes reais mostrando
    legenda/corte/detalhe em vez do carimbo de título. Agora a
    verificação por conteúdo exige uma pontuação mínima (ver
    LIMIAR_MINIMO_CONTEUDO_VERIFICADO) e rejeita esses casos."""
    from scanner.detector_carimbo import detectar_carimbo_verificado

    imagem = np.full((1200, 1600, 3), 255, dtype=np.uint8)
    for (x0, y0, x1, y1) in [(50, 50, 500, 400), (1100, 800, 1580, 1150)]:
        cv2.rectangle(imagem, (x0, y0), (x1, y1), (0, 0, 0), 2)
        for y in range(y0 + 20, y1 - 10, 12):
            cv2.line(imagem, (x0 + 10, y), (x1 - 10, y), (0, 0, 0), 1)

    class _ResultadoOCRFalso:
        def __init__(self, texto):
            self.texto = texto

    def _ocr_sem_palavra_chave(imagem, idiomas):
        # texto denso e genérico, sem palavra-chave real de carimbo
        # nem telefone — igual ao que apareceu nos recortes reais
        return _ResultadoOCRFalso("BLOCOS DE CONCRETO JUNTAS A PRUMO APARENTES " * 3)

    resultado = detectar_carimbo_verificado(imagem, _ocr_sem_palavra_chave)
    assert resultado is None


def test_verificacao_por_conteudo_recorta_da_resolucao_original_nao_da_busca_reduzida():
    """Reproduz um segundo problema real, causado pela correção de
    performance anterior: reduzir a PÁGINA INTEIRA antes de recortar
    cada candidato deixava o texto do candidato pequeno/borrado demais
    para o OCR reconhecer — inclusive em carimbos de verdade, fazendo
    a taxa de detecção despencar quase a zero. Agora só a busca por
    contorno roda na cópia reduzida; o recorte de cada candidato (e o
    resultado final) vem da imagem em resolução ORIGINAL. Confirma que
    os recortes entregues ao OCR chegam próximos do teto de
    verificação (1400px), não achatados pela redução da página."""
    from scanner.detector_carimbo import detectar_carimbo_verificado, TAMANHO_MAX_VERIFICACAO

    imagem = np.full((10000, 14000, 3), 255, dtype=np.uint8)
    cv2.rectangle(imagem, (100, 100), (3000, 2500), (0, 0, 0), 4)
    for y in range(150, 2400, 30):
        cv2.line(imagem, (150, y), (2950, y), (0, 0, 0), 2)
    cv2.rectangle(imagem, (11000, 7500), (13900, 9900), (0, 0, 0), 4)
    for y in range(7550, 9850, 30):
        cv2.line(imagem, (11050, y), (13850, y), (0, 0, 0), 2)

    class _ResultadoOCRFalso:
        def __init__(self, texto):
            self.texto = texto

    tamanhos_vistos = []

    def _ocr_registra_tamanho(recorte, idiomas):
        tamanhos_vistos.append(recorte.shape[:2])
        return _ResultadoOCRFalso("")

    detectar_carimbo_verificado(imagem, _ocr_registra_tamanho)

    assert tamanhos_vistos, "esperava pelo menos uma verificação por OCR"
    for altura, largura in tamanhos_vistos:
        assert max(altura, largura) >= TAMANHO_MAX_VERIFICACAO * 0.7, (
            f"recorte chegou pequeno demais ao OCR: {(altura, largura)} — "
            "sinal de que veio da página já reduzida, não da resolução original"
        )


def test_pipeline_classificacao_end_to_end_com_texto_simulado():
    texto_ocr_simulado = "IMPLANTAÇÃO GERAL\nESCALA 1:200\nARQUITETO: CARLOS BARJAS MILLAN"
    resultado = classificar(texto_ocr_simulado)
    assert resultado.tipo == "Implantação"


class _AutoFalsoComFallback:
    """Duck-type mínimo: só precisa do atributo que
    _tentar_fallback_geometrico usa, sem instanciar o
    PipelineProcessamento inteiro (que depende de SQLAlchemy)."""
    def __init__(self, fallback_fn):
        self.detector_carimbo_fallback_fn = fallback_fn


def test_fallback_geometrico_nao_e_chamado_quando_nao_configurado():
    from scanner.pipeline import PipelineProcessamento

    auto_falso = _AutoFalsoComFallback(fallback_fn=None)
    imagem = _criar_imagem_sintetica_com_carimbo()

    resultado = PipelineProcessamento._tentar_fallback_geometrico(auto_falso, imagem)
    assert resultado is None


def test_fallback_geometrico_usa_a_funcao_fornecida_quando_o_primario_falha():
    from scanner.pipeline import PipelineProcessamento
    from scanner.detector_carimbo import CarimboDetectado

    chamadas = []

    def fallback_fn(imagem):
        chamadas.append(imagem)
        return CarimboDetectado(x=10, y=10, largura=100, altura=50, confianca=0.7, canto="inferior_direito")

    auto_falso = _AutoFalsoComFallback(fallback_fn=fallback_fn)
    imagem = _criar_imagem_sintetica_com_carimbo()

    resultado = PipelineProcessamento._tentar_fallback_geometrico(auto_falso, imagem)

    assert resultado is not None
    assert resultado.canto == "inferior_direito"
    assert len(chamadas) == 1  # uma única tentativa, não uma por orientação


def test_fallback_geometrico_passa_a_imagem_em_resolucao_original_sem_reduzir():
    """A redução (pra busca rápida) e o recorte em resolução original
    (pra verificação) agora acontecem DENTRO de
    detectar_carimbo_verificado — o pipeline só repassa a imagem como
    recebeu, sem mexer na resolução."""
    from scanner.pipeline import PipelineProcessamento

    imagem_grande = np.full((9000, 12000, 3), 255, dtype=np.uint8)
    tamanhos_recebidos = []

    def fallback_fn(imagem):
        tamanhos_recebidos.append(imagem.shape[:2])
        return None

    auto_falso = _AutoFalsoComFallback(fallback_fn=fallback_fn)
    PipelineProcessamento._tentar_fallback_geometrico(auto_falso, imagem_grande)

    assert tamanhos_recebidos == [(9000, 12000)]


def test_fallback_geometrico_devolve_none_se_a_2a_estrategia_tambem_falhar():
    from scanner.pipeline import PipelineProcessamento

    auto_falso = _AutoFalsoComFallback(fallback_fn=lambda imagem: None)
    imagem = _criar_imagem_sintetica_com_carimbo()

    resultado = PipelineProcessamento._tentar_fallback_geometrico(auto_falso, imagem)
    assert resultado is None


class _AutoFalsoOrientacao:
    """Duck-type pra testar _detectar_em_qualquer_orientacao sem
    instanciar o PipelineProcessamento inteiro (depende de SQLAlchemy).
    Simula o detector primário (modelo treinado) falhando em TODAS as
    orientações — o cenário onde a 2ª estratégia vira a única chance."""
    deteccao_multiorientacao = True

    def __init__(self, fallback_fn):
        self.detector_carimbo_fn = lambda imagem: None  # modelo treinado nunca acha nada
        self.detector_carimbo_fallback_fn = fallback_fn

    def _tentar_fallback_geometrico(self, imagem):
        from scanner.pipeline import PipelineProcessamento
        return PipelineProcessamento._tentar_fallback_geometrico(self, imagem)


def test_quando_modelo_falha_em_tudo_testa_as_4_rotacoes_com_fallback_completo():
    """Reproduz o problema real: quando o modelo treinado não acha
    nada em NENHUMA das 8 orientações testadas, a busca antiga
    confiava só num palpite de orientação genérico (`corrigir_orientacao`,
    um heurístico de OCR não específico de carimbo) para a 2ª
    estratégia — se esse palpite errasse a rotação, a 2ª estratégia
    buscava nos cantos errados e não achava nada. Confirmado num
    arquivo real: 0.87 de confiança na orientação certa, só 0.36
    (abaixo do limiar) na orientação escolhida pelo palpite genérico.

    Agora a 2ª estratégia é testada nas 4 rotações básicas, com saída
    antecipada assim que uma cruza CONFIANCA_BOA_O_BASTANTE."""
    from scanner.pipeline import PipelineProcessamento
    from scanner.detector_carimbo import CarimboDetectado

    imagem = _criar_imagem_sintetica_com_carimbo()
    chamadas = []

    def fallback_fn(imagem_testada):
        chamadas.append(imagem_testada)
        # simula: só a 3ª rotação testada (índice 2) "parece" um carimbo de verdade
        if len(chamadas) == 3:
            return CarimboDetectado(x=5, y=5, largura=50, altura=30, confianca=0.87, canto="inferior_direito")
        return None

    auto_falso = _AutoFalsoOrientacao(fallback_fn=fallback_fn)
    imagem_final, carimbo = PipelineProcessamento._detectar_em_qualquer_orientacao(auto_falso, imagem)

    assert carimbo is not None
    assert carimbo.confianca == 0.87
    assert len(chamadas) == 3  # parou assim que achou, não testou a 4ª


def test_quando_nenhuma_rotacao_acha_nada_devolve_none_sem_travar():
    from scanner.pipeline import PipelineProcessamento

    imagem = _criar_imagem_sintetica_com_carimbo()
    auto_falso = _AutoFalsoOrientacao(fallback_fn=lambda imagem: None)

    imagem_final, carimbo = PipelineProcessamento._detectar_em_qualquer_orientacao(auto_falso, imagem)

    assert carimbo is None
    assert imagem_final is not None  # ainda devolve alguma imagem (orientação corrigida por OCR)
