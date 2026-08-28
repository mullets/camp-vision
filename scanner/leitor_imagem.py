"""
scanner/leitor_imagem.py
=========================
Leitura de arquivos TIFF (incluindo multipágina), verificação de
resolução, correção automática de orientação, melhoria de contraste
e redução de ruído.

Toda a interface é baseada em arrays NumPy no formato OpenCV (BGR),
para que o restante do pipeline (detecção de carimbo, OCR) possa
trabalhar de forma homogênea independentemente da fonte da imagem.
Isso também facilita a expansão futura para outros formatos de
entrada, como PDF: basta implementar um novo `carregar_paginas`.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import tifffile

logger = logging.getLogger("campvision.scanner")

# Margem que a melhor orientação precisa ter sobre a original para
# valer a correção. São duas porque os casos são diferentes:
#
# - PÁGINA INTEIRA: uma prancha densa, cheia de cotas e textos em
#   várias direções, produz pontuações quase empatadas entre as 8
#   combinações. Sem uma margem folgada, uma prancha JÁ CORRETA
#   acabava girada por engano (aconteceu de verdade).
# - RECORTE DO CARIMBO: é quase só texto correndo numa direção. A
#   orientação certa vence de forma consistente, mas às vezes por
#   pouco (num carimbo real de paisagismo a vencedora deu 1,12x e
#   ficava de fora, saindo deitada). Aqui uma margem menor é segura.
MARGEM_PAGINA_INTEIRA = 1.20
MARGEM_RECORTE_CARIMBO = 1.05

RESOLUCAO_MINIMA_DPI = 150  # abaixo disso, OCR tende a degradar muito

# Piso em pixels absolutos, usado só quando o arquivo não tem DPI
# gravado (comum em JPG) e por isso a checagem por DPI não pode
# rodar. Não substitui o DPI — é só uma rede de segurança best-effort
# para pegar o caso óbvio de um scan pequeno demais.
RESOLUCAO_MINIMA_PIXELS_SEM_DPI = 2000


@dataclass
class ImagemCarregada:
    """Representa uma página de imagem já carregada e pronta para o pipeline."""

    caminho_origem: Path
    indice_pagina: int
    imagem: np.ndarray  # BGR, uint8
    dpi: tuple[float, float] | None


def _extrair_dpi(caminho: Path) -> tuple[float, float] | None:
    """Tenta extrair a resolução (DPI) do TIFF a partir de suas tags."""
    try:
        with tifffile.TiffFile(str(caminho)) as tif:
            page = tif.pages[0]
            x_res = page.tags.get("XResolution")
            y_res = page.tags.get("YResolution")
            if x_res and y_res:
                xr = x_res.value[0] / x_res.value[1] if isinstance(x_res.value, tuple) else float(x_res.value)
                yr = y_res.value[0] / y_res.value[1] if isinstance(y_res.value, tuple) else float(y_res.value)
                return (xr, yr)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Não foi possível ler DPI de %s: %s", caminho, exc)
    return None


EXTENSOES_TIFF = {".tif", ".tiff"}
EXTENSOES_IMAGEM_SIMPLES = {".jpg", ".jpeg", ".png", ".bmp"}
EXTENSOES_PDF = {".pdf"}

# Resolução de renderização de PDFs. 300 DPI é o suficiente para ler
# texto de carimbo; abaixo disso o OCR começa a errar em letra
# pequena.
DPI_RENDERIZACAO_PDF = 300

# Teto de tamanho ao renderizar. Pranchas de arquitetura em PDF são
# frequentemente A0 ou maiores: a 300 DPI, uma A0 daria ~9900x14000 px
# (mais de 400 MB em memória por página). O teto mantém resolução
# suficiente para o carimbo sem arriscar estourar a RAM.
LADO_MAXIMO_PDF = 8000

# A biblioteca pdfium NÃO é segura para uso simultâneo por várias
# threads. Como o lote processa arquivos em paralelo, sem esta trava
# duas threads abriam/renderizavam PDFs ao mesmo tempo e o processo
# inteiro morria com segmentation fault (aconteceu num lote de 196
# PDFs). A renderização é serializada; o resto do pipeline (OCR,
# detecção) continua paralelo.
_trava_pdf = threading.Lock()


def carregar_paginas(caminho: Path) -> Iterator[ImagemCarregada]:
    """Carrega todas as páginas de um arquivo de imagem, retornando um
    iterador de ImagemCarregada.

    TIFF (possivelmente multipágina) usa `tifffile`, que trata casos
    específicos desse formato (bilevel/1-bit, múltiplas páginas,
    fotometria invertida). Outros formatos comuns (JPG, PNG, BMP) são
    sempre de página única e usam um carregamento mais simples via
    OpenCV. PDFs são rasterizados página a página (ver
    `_carregar_paginas_pdf`)."""
    extensao = caminho.suffix.lower()
    if extensao in EXTENSOES_TIFF:
        yield from _carregar_paginas_tiff(caminho)
    elif extensao in EXTENSOES_IMAGEM_SIMPLES:
        yield from _carregar_pagina_simples(caminho)
    elif extensao in EXTENSOES_PDF:
        yield from _carregar_paginas_pdf(caminho)
    else:
        raise ValueError(f"Formato de arquivo não suportado: {extensao}")


def _carregar_paginas_tiff(caminho: Path) -> Iterator[ImagemCarregada]:
    dpi = _extrair_dpi(caminho)
    with tifffile.TiffFile(str(caminho)) as tif:
        for indice, page in enumerate(tif.pages):
            try:
                array = page.asarray()
            except Exception as exc:  # noqa: BLE001
                if "imagecodecs" in str(exc):
                    raise RuntimeError(
                        f"Não foi possível ler '{caminho.name}': este TIFF usa um tipo de "
                        f"compressão (ex.: LZW) que exige o pacote 'imagecodecs', que não "
                        f"está instalado. Rode 'pip install imagecodecs' no ambiente do "
                        f"CAMP Vision e processe o lote de novo."
                    ) from exc
                raise
            fotometria_minima_e_branco = int(page.photometric) == int(tifffile.PHOTOMETRIC.MINISWHITE)
            imagem_bgr = _normalizar_para_bgr(array, fotometria_minima_e_branco)
            if dpi and (dpi[0] < RESOLUCAO_MINIMA_DPI or dpi[1] < RESOLUCAO_MINIMA_DPI):
                logger.warning(
                    "Resolução baixa (%.0fx%.0f DPI) em %s (página %d)",
                    dpi[0], dpi[1], caminho.name, indice,
                )
            yield ImagemCarregada(
                caminho_origem=caminho,
                indice_pagina=indice,
                imagem=imagem_bgr,
                dpi=dpi,
            )


def _carregar_paginas_pdf(caminho: Path) -> Iterator[ImagemCarregada]:
    """Rasteriza cada página do PDF para uma imagem, para que o
    restante do pipeline (detecção de carimbo, OCR) funcione sem
    saber que a origem era um PDF.

    Usamos `pypdfium2` por dois motivos práticos: tem wheel pronta
    para Mac Intel (as máquinas do acervo são antigas) e não depende
    de nada instalado no sistema — ao contrário do `pdf2image`, que
    exige o poppler.

    Observação: PDFs de arquivo costumam conter apenas a IMAGEM
    digitalizada da prancha, sem camada de texto. Mesmo quando têm
    texto embutido, rasterizar e passar pelo mesmo caminho das outras
    origens mantém um comportamento único — e o carimbo continua
    sendo localizado visualmente, que é o que o modelo treinado faz."""
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise ValueError(
            "Leitura de PDF requer o pacote 'pypdfium2' (pip install pypdfium2)."
        ) from exc

    # Cada página é renderizada dentro da trava, e a imagem já
    # convertida para numpy sai dela — assim nenhuma estrutura do
    # pdfium é tocada fora da região serializada.
    indice = 0
    while True:
        with _trava_pdf:
            documento = pdfium.PdfDocument(str(caminho))
            try:
                if indice >= len(documento):
                    break
                pagina = documento[indice]
                largura_pt, altura_pt = pagina.get_size()
                escala = DPI_RENDERIZACAO_PDF / 72
                maior_lado = max(largura_pt, altura_pt) * escala
                if maior_lado > LADO_MAXIMO_PDF:
                    escala *= LADO_MAXIMO_PDF / maior_lado
                    logger.info(
                        "Página %d de %s é muito grande; renderizando a %.0f DPI em vez de %d.",
                        indice + 1, caminho.name, escala * 72, DPI_RENDERIZACAO_PDF,
                    )
                imagem_bgr = cv2.cvtColor(pagina.render(scale=escala).to_numpy(), cv2.COLOR_RGB2BGR)
            finally:
                documento.close()

        yield ImagemCarregada(
            caminho_origem=caminho,
            indice_pagina=indice,
            imagem=imagem_bgr,
            dpi=(escala * 72, escala * 72),
        )
        indice += 1


def _carregar_pagina_simples(caminho: Path) -> Iterator[ImagemCarregada]:
    """Carrega um JPG/PNG/BMP — sempre uma única 'página', sem as
    particularidades de TIFF (bilevel, multipágina)."""
    imagem = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if imagem is None:
        raise ValueError(f"Não foi possível abrir a imagem: {caminho}")

    dpi = None
    try:
        from PIL import Image
        with Image.open(caminho) as img_pil:
            info_dpi = img_pil.info.get("dpi")
            if info_dpi:
                dpi = (float(info_dpi[0]), float(info_dpi[1]))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Não foi possível ler DPI de %s: %s", caminho, exc)

    if dpi and (dpi[0] < RESOLUCAO_MINIMA_DPI or dpi[1] < RESOLUCAO_MINIMA_DPI):
        logger.warning("Resolução baixa (%.0fx%.0f DPI) em %s", dpi[0], dpi[1], caminho.name)
    elif dpi is None:
        # JPG raramente grava DPI de verdade (muitos scanners/exportações
        # não embutem essa tag) — sem isso, um scan pequeno demais nunca
        # gerava aviso nenhum, já que a checagem acima só roda quando
        # `dpi` existe. Sem saber o tamanho físico do papel original não
        # dá pra calcular um DPI equivalente, mas um piso em pixels
        # absolutos pega o caso óbvio: pranchas de arquitetura (A0/A1)
        # digitalizadas decentemente não ficam abaixo de ~2000px no lado
        # menor, então isso serve como sinal de alerta best-effort.
        altura, largura = imagem.shape[:2]
        if min(altura, largura) < RESOLUCAO_MINIMA_PIXELS_SEM_DPI:
            logger.warning(
                "Resolução possivelmente baixa em %s (%dx%d px, sem DPI gravado no arquivo "
                "para calcular a resolução real) — confira se o scan não ficou pequeno demais.",
                caminho.name, largura, altura,
            )

    yield ImagemCarregada(
        caminho_origem=caminho,
        indice_pagina=0,
        imagem=imagem,
        dpi=dpi,
    )


def _normalizar_para_bgr(array: np.ndarray, fotometria_minima_e_branco: bool = False) -> np.ndarray:
    """Converte arrays de diferentes profundidades/formatos de canal
    (bilevel/1-bit, escala de cinza, RGB, RGBA, 16 bits) para BGR
    uint8, formato padrão usado pelo restante do pipeline (OpenCV).

    IMPORTANTE — polaridade fotométrica: TIFFs bilevel (1 bit) muito
    comuns em digitalizações de desenhos técnicos usam a convenção
    "MinIsWhite" (photometric=0), em que o valor 0 representa branco
    e o valor 1 (máximo) representa preto — o INVERSO da convenção
    "MinIsBlack" mais comum em imagens em geral. Sem levar isso em
    conta, o desenho (tinta) vira branco e o fundo vira preto,
    invertendo completamente a imagem e quebrando toda a detecção de
    contornos/carimbo que segue no pipeline."""
    if array.dtype == np.bool_:
        if fotometria_minima_e_branco:
            # MinIsWhite: True (bit 1) = preto/tinta, False (bit 0) = branco/fundo
            array = np.where(array, 0, 255).astype(np.uint8)
        else:
            # MinIsBlack (padrão mais comum): True = branco, False = preto
            array = array.astype(np.uint8) * 255
    elif array.dtype != np.uint8:
        # Normaliza imagens de 16 bits (comuns em digitalizações de alta qualidade)
        array = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if fotometria_minima_e_branco:
            array = 255 - array

    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.ndim == 3 and array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
    if array.ndim == 3 and array.shape[2] == 3:
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    raise ValueError(f"Formato de imagem não suportado: shape={array.shape}, dtype={array.dtype}")


def _pontuacao_ocr(imagem_gray: np.ndarray) -> float:
    """Roda OCR e retorna a SOMA das confianças das palavras
    reconhecidas — usado como sinal de "isto está com a orientação
    correta?". Somar (em vez de tirar a média) recompensa tanto
    reconhecer mais palavras quanto reconhecê-las com mais confiança;
    usar só a média deixaria uma única palavra "sortuda" empatar com
    uma orientação que reconhece o texto inteiro corretamente."""
    import pytesseract

    try:
        dados = pytesseract.image_to_data(imagem_gray, output_type=pytesseract.Output.DICT, config="--psm 3")
    except Exception as exc:  # noqa: BLE001
        logger.debug("OCR de teste de orientação falhou: %s", exc)
        return 0.0

    confiancas = [float(c) for c in dados.get("conf", []) if c not in ("-1", -1)]
    return sum(confiancas)


def _gerar_transformacoes(imagem: np.ndarray) -> dict[tuple[int, bool], np.ndarray]:
    """Gera as 8 combinações possíveis de rotação (0/90/180/270°) e
    espelhamento horizontal — algumas pranchas do acervo testado
    vieram espelhadas (comum em fotocópias antigas mal alimentadas no
    equipamento), não só rotacionadas, então rotação sozinha não
    cobre todos os casos reais observados."""
    rotacoes = {
        0: imagem,
        90: cv2.rotate(imagem, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(imagem, cv2.ROTATE_180),
        270: cv2.rotate(imagem, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }
    transformacoes = {}
    for angulo, img_rotacionada in rotacoes.items():
        transformacoes[(angulo, False)] = img_rotacionada
        transformacoes[(angulo, True)] = cv2.flip(img_rotacionada, 1)
    return transformacoes


def _detectar_transformacao_por_confianca_ocr(
    imagem: np.ndarray, tamanho_max_teste: int = 1400, margem_minima: float = MARGEM_PAGINA_INTEIRA,
) -> tuple[int, bool]:
    """Detecta a transformação correta da prancha (rotação em
    0°/90°/180°/270°, combinada ou não com espelhamento horizontal)
    rodando OCR real numa versão reduzida da imagem em cada uma das
    oito combinações possíveis e comparando a pontuação — a
    combinação correta produz reconhecimento de texto bem mais
    confiável que as demais.

    Preferimos este método a confiar no OSD (Orientation and Script
    Detection) do Tesseract: testamos e o OSD se mostrou pouco
    confiável para a letra tipo carimbo técnico (toda maiúscula,
    estêncil) usada em pranchas de arquitetura — chegou a classificar
    o texto como grego. Rodar o OCR de verdade e comparar a pontuação
    é mais lento (8 chamadas em vez de 1), mas mede exatamente o que
    importa — testes no acervo real mostraram pranchas de cabeça para
    baixo, de lado, e até espelhadas (fotocópias antigas).

    Retorna (ângulo, espelhar) — ver `_aplicar_transformacao`.
    """
    altura, largura = imagem.shape[:2]
    escala = min(1.0, tamanho_max_teste / max(altura, largura))
    pequena = cv2.resize(imagem, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA) if escala < 1.0 else imagem
    cinza = cv2.cvtColor(pequena, cv2.COLOR_BGR2GRAY)

    candidatos = _gerar_transformacoes(cinza)
    pontuacoes = {chave: _pontuacao_ocr(img_teste) for chave, img_teste in candidatos.items()}
    logger.debug("Teste de orientação (pontuações OCR): %s", pontuacoes)

    vencedor = max(pontuacoes, key=pontuacoes.get)
    pontuacao_vencedora = pontuacoes[vencedor]
    pontuacao_original = pontuacoes[(0, False)]

    # Só aplica uma correção se houver vantagem CLARA sobre a
    # orientação original — em pranchas cheias de cotas e textos em
    # várias direções (não um recorte de carimbo com texto corrido),
    # a diferença de pontuação entre as 8 combinações costuma ser
    # pequena (ruído de OCR, não sinal real de orientação errada).
    # Testes reais mostraram: a orientação certa de um carimbo
    # recortado vence por quase o dobro; numa prancha inteira densa,
    # todas as combinações às vezes ficam a poucos % uma da outra —
    # nesse caso é mais seguro não mexer do que arriscar girar/
    # espelhar uma prancha que já estava correta.
    if pontuacao_vencedora <= 0:
        return (0, False)  # nenhuma orientação achou texto reconhecível — não mexe
    if pontuacao_original <= 0:
        return vencedor  # original não achou nada, mas outra orientação achou — sinal real
    if pontuacao_vencedora < pontuacao_original * margem_minima:
        return (0, False)  # vantagem pequena demais — pode ser só ruído de OCR

    return vencedor


def _aplicar_transformacao(imagem: np.ndarray, angulo: int, espelhar: bool) -> np.ndarray:
    resultado = _rotacionar(imagem, angulo)
    if espelhar:
        resultado = cv2.flip(resultado, 1)
    return resultado


def _rotacionar(imagem: np.ndarray, angulo: int) -> np.ndarray:
    if angulo == 90:
        return cv2.rotate(imagem, cv2.ROTATE_90_CLOCKWISE)
    if angulo == 180:
        return cv2.rotate(imagem, cv2.ROTATE_180)
    if angulo == 270:
        return cv2.rotate(imagem, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return imagem


def corrigir_orientacao(imagem: np.ndarray, margem_minima: float = MARGEM_PAGINA_INTEIRA) -> np.ndarray:
    """Corrige automaticamente a orientação da prancha — rotação
    (0°/90°/180°/270°) e/ou espelhamento horizontal — comparando a
    confiança real do OCR nas oito combinações possíveis (ver
    `_detectar_transformacao_por_confianca_ocr`). O acervo testado
    tinha pranchas de cabeça para baixo, de lado e espelhadas."""
    angulo, espelhar = _detectar_transformacao_por_confianca_ocr(imagem, margem_minima=margem_minima)
    if angulo or espelhar:
        logger.info("Orientação da prancha corrigida (rotação=%d°, espelhado=%s).", angulo, espelhar)
        return _aplicar_transformacao(imagem, angulo, espelhar)
    return imagem


def reduzir_para_ocr(imagem: np.ndarray, tamanho_max: int = 2500) -> np.ndarray:
    """Reduz uma imagem para um tamanho máximo antes de rodar OCR.

    Pranchas de arquitetura digitalizadas costumam ser enormes (mais
    de 10000px de lado) — rodar Tesseract em resolução original em
    recortes desse tamanho pode levar de 30 segundos a mais de um
    minuto POR ARQUIVO, mesmo já sendo só o recorte do carimbo (não a
    prancha inteira). Tesseract não precisa de resolução tão alta
    para reconhecer texto de tamanho normal; reduzir para um teto
    razoável acelera bastante sem perda de precisão perceptível."""
    altura, largura = imagem.shape[:2]
    maior_lado = max(altura, largura)
    if maior_lado <= tamanho_max:
        return imagem
    escala = tamanho_max / maior_lado
    nova_dimensao = (max(1, int(largura * escala)), max(1, int(altura * escala)))
    return cv2.resize(imagem, nova_dimensao, interpolation=cv2.INTER_AREA)


def realcar_para_ocr(imagem: np.ndarray) -> np.ndarray:
    """Versão de alto contraste do recorte, para tentar o OCR quando a
    digitalização é fraca.

    Muitas pranchas de acervo são traço claro sobre papel vegetal ou
    cópias desbotadas: o texto existe, mas com contraste baixo demais
    para o Tesseract. CLAHE (equalização local) seguido de
    binarização de Otsu recupera boa parte desse texto — num carimbo
    real de 1953, foi a diferença entre não ler nada e ler
    "ENGENHEIRO ARQUITETO".

    Não substitui o recorte original: realce agressivo pode PIORAR
    digitalizações limpas, então o pipeline tenta as duas versões e
    fica com a que produzir a melhor leitura."""
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) if imagem.ndim == 3 else imagem
    realcada = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(cinza)
    _, binarizada = cv2.threshold(realcada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binarizada, cv2.COLOR_GRAY2BGR)


def melhorar_contraste(imagem: np.ndarray) -> np.ndarray:
    """Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization)
    no canal de luminância para melhorar o contraste sem estourar
    áreas já claras — importante em pranchas desbotadas ou com
    iluminação irregular na digitalização."""
    lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def reduzir_ruido(imagem: np.ndarray) -> np.ndarray:
    """Reduz ruído preservando bordas (essencial para não borrar
    texto fino de carimbos e cotas), usando filtro bilateral."""
    return cv2.bilateralFilter(imagem, d=5, sigmaColor=50, sigmaSpace=50)


def pre_processar(imagem: np.ndarray) -> np.ndarray:
    """Pipeline completo de pré-processamento aplicado a cada página
    antes da detecção de carimbo e do OCR."""
    imagem = corrigir_orientacao(imagem)
    imagem = reduzir_ruido(imagem)
    imagem = melhorar_contraste(imagem)
    return imagem
