"""
bench/avaliar.py
==================
Avaliador objetivo do CAMP Vision.

Roda o pipeline completo sobre uma pasta de pranchas e compara o
resultado com um GABARITO preenchido à mão, produzindo uma nota por
campo. Serve para responder com números — e não "no olho" — perguntas
como:

    "Essa mudança melhorou ou piorou a leitura da folha?"
    "O modelo treinado é melhor que a heurística NESTE acervo?"

Sem isso, cada ajuste é uma aposta: olha-se meia dúzia de recortes e
tenta-se adivinhar se melhorou. Com isso, cada ajuste vira uma
comparação de placar antes/depois.

SEGURANÇA: a avaliação NUNCA renomeia, move ou altera os arquivos
avaliados — renomeação e arquivamento são desligados à força, e a
saída vai para uma pasta temporária.

Uso:
    python -m bench.avaliar --pasta <pasta_das_pranchas> \\
                            --gabarito bench/gabarito.csv \\
                            [--modelo caminho/best.pt] \\
                            [--confianca 0.3]

O gabarito é um CSV com uma linha por prancha (só as colunas que
você quiser cobrar; deixe em branco o que não for avaliar):

    arquivo,carimbo,folha,tipo,escala,ano,arquiteto,projeto,endereco
    Cópia de DEST2696.tif,sim,1,Planta,1:250,1974,Sami Bussab,,RUA PARAMU
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from ai.fallback_regras import extrair_por_regras  # noqa: E402
from classificacao.classificador import classificar  # noqa: E402
from ocr.tesseract_ocr import extrair_texto  # noqa: E402
from scanner.detector_carimbo import _contar_acertos_aproximados, criar_detector  # noqa: E402
from scanner.folha_visual import ler_numero_folha  # noqa: E402
from scanner.leitor_imagem import (  # noqa: E402
    _rotacionar, carregar_paginas, corrigir_orientacao, reduzir_para_ocr,
)
from scanner.propagacao import propagar_metadados_projeto  # noqa: E402

logger = logging.getLogger("campvision.bench")

# Campos avaliados e como cada um é comparado:
#   "exato"   — precisa bater igual (após normalizar)
#   "contem"  — acerta se o valor esperado aparece no obtido (ou
#               vice-versa); usado em texto livre, onde o OCR
#               costuma trazer algo a mais ou a menos
CAMPOS_AVALIADOS = {
    "folha": "exato",
    "tipo": "exato",
    "escala": "exato",
    "ano": "exato",
    "arquiteto": "contem",
    "projeto": "contem",
    "endereco": "contem",
}

# Nome da coluna do CSV de saída correspondente a cada campo do gabarito.
ATRIBUTO = {
    "folha": "numero",
    "tipo": "tipo",
    "escala": "escala",
    "ano": "ano",
    "arquiteto": "arquiteto",
    "projeto": "projeto",
    "endereco": "endereco",
}


# Mínimo de palavras de vocabulário de carimbo para considerar que o
# carimbo foi localizado E lido de forma útil.
ACERTOS_MINIMOS_CARIMBO_LIDO = 2


def normalizar(valor: str) -> str:
    """Normaliza para comparação: sem acento, sem pontuação de borda,
    maiúsculas, espaços colapsados."""
    if not valor:
        return ""
    texto = unicodedata.normalize("NFKD", valor).encode("ascii", "ignore").decode("ascii")
    return " ".join(texto.upper().split()).strip(" .,;:-")


@dataclass
class PlacarCampo:
    acertos: int = 0
    erros: int = 0
    cobrados: int = 0  # linhas do gabarito que preencheram este campo
    exemplos_erro: list = field(default_factory=list)

    @property
    def taxa(self) -> float:
        return self.acertos / self.cobrados if self.cobrados else 0.0


def carregar_gabarito(caminho: Path) -> dict[str, dict]:
    with open(caminho, encoding="utf-8-sig") as arquivo:
        return {linha["arquivo"].strip(): linha for linha in csv.DictReader(arquivo)}


@dataclass
class RegistroAvaliado:
    """Registro mínimo compatível com o que a propagação espera —
    evita arrastar o exportador (e suas dependências) para dentro do
    avaliador."""
    arquivo: str
    arquivo_original: str
    projeto: str = ""
    cliente: str = ""
    arquiteto: str = ""
    cidade: str = ""
    endereco: str = ""
    ano: str = ""
    prancha: str = ""
    numero: str = ""
    escala: str = ""
    tipo: str = ""
    fase: str = ""
    observacoes: str = ""
    # Texto lido do carimbo, guardado ANTES da propagação: é a única
    # evidência honesta de que o carimbo foi de fato localizado e
    # lido. Medir "carimbo localizado" por campos preenchidos seria
    # enganoso, porque a propagação preenche campos a partir de
    # outras pranchas.
    texto_ocr: str = ""


def processar(pasta: Path, pasta_saida: Path, modelo: str | None, confianca: float,
              idiomas: list[str]) -> dict[str, RegistroAvaliado]:
    """Roda a cadeia de análise (sem banco de dados e sem tocar nos
    arquivos) e devolve um registro por prancha.

    Deliberadamente NÃO usa o ProcessadorLote: o avaliador precisa
    medir a qualidade da ANÁLISE (localizar carimbo, ler, extrair,
    classificar), não a infraestrutura de renomeação/arquivamento/
    banco — que além de irrelevante aqui, alteraria os arquivos
    avaliados."""
    detector = criar_detector(
        modo="modelo_treinado" if modelo else "heuristico",
        caminho_modelo=modelo,
        confianca_minima=confianca,
        ocr_fn=extrair_texto,
        idiomas_ocr=idiomas,
        tamanho_imagem_ml=0,  # detectado do próprio modelo
    )

    pasta_saida.mkdir(parents=True, exist_ok=True)
    registros: dict[str, RegistroAvaliado] = {}

    arquivos = sorted(
        caminho for caminho in pasta.iterdir()
        if caminho.suffix.lower() in {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".pdf"}
    )

    for indice, caminho in enumerate(arquivos, start=1):
        print(f"  [{indice}/{len(arquivos)}] {caminho.name}", flush=True)
        registro = RegistroAvaliado(arquivo=caminho.name, arquivo_original=caminho.name)
        try:
            pagina = next(iter(carregar_paginas(caminho)))
            # Procura o carimbo em todas as orientações (o detector
            # treinado não é invariante a rotação/espelhamento) e fica
            # com a de maior confiança — mesma lógica do pipeline.
            # Busca a orientação numa cópia REDUZIDA (ver pipeline:
            # testar 8 orientações em resolução total estoura a RAM) e
            # só aplica a vencedora em resolução total.
            imagem, carimbo = None, None
            if modelo:
                alt, lar = pagina.imagem.shape[:2]
                escala = min(1.0, 1600 / max(alt, lar))
                pequena = (
                    cv2.resize(pagina.imagem, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
                    if escala < 1.0 else pagina.imagem
                )
                melhor, melhor_conf = None, -1.0
                for espelhar in (False, True):
                    base = cv2.flip(pequena, 1) if espelhar else pequena
                    for angulo in (0, 90, 180, 270):
                        achado = detector(_rotacionar(base, angulo))
                        if achado is not None and achado.confianca > melhor_conf:
                            melhor, melhor_conf = (angulo, espelhar), achado.confianca
                    if melhor_conf >= 0.60:
                        break
                if melhor is not None:
                    angulo, espelhar = melhor
                    imagem = _rotacionar(cv2.flip(pagina.imagem, 1) if espelhar else pagina.imagem, angulo)
                    carimbo = detector(imagem)
            else:
                imagem = corrigir_orientacao(pagina.imagem)
                carimbo = detector(imagem)

            if carimbo is None:
                imagem = corrigir_orientacao(pagina.imagem)

            texto = ""
            if carimbo is not None:
                recorte = corrigir_orientacao(carimbo.recortar(imagem))
                cv2.imwrite(str(pasta_saida / f"{caminho.stem}_carimbo.png"), recorte)
                texto = extrair_texto(reduzir_para_ocr(recorte), idiomas).texto

                metadados = extrair_por_regras(texto)
                if not metadados.numero:
                    metadados.numero = ler_numero_folha(recorte) or ""

                registro.projeto = metadados.projeto
                registro.cliente = metadados.cliente
                registro.arquiteto = metadados.arquiteto
                registro.cidade = metadados.cidade
                registro.endereco = metadados.endereco
                registro.ano = metadados.ano
                registro.numero = metadados.numero
                registro.escala = metadados.escala
                registro.fase = metadados.fase
                registro.tipo = classificar(texto).tipo

            registro.texto_ocr = texto
            (pasta_saida / f"{caminho.stem}_ocr.txt").write_text(texto, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falha ao processar %s: %s", caminho.name, exc)

        registros[caminho.name] = registro

    propagar_metadados_projeto(list(registros.values()))
    return registros


def comparar(gabarito: dict[str, dict], resultado: dict) -> tuple[dict, int, int]:
    placares = {campo: PlacarCampo() for campo in CAMPOS_AVALIADOS}
    carimbo_esperado = 0
    carimbo_encontrado = 0

    for nome_arquivo, esperado in gabarito.items():
        obtido = resultado.get(nome_arquivo)
        if obtido is None:
            logger.warning("Arquivo do gabarito não encontrado no resultado: %s", nome_arquivo)
            continue

        if normalizar(esperado.get("carimbo", "")) in ("SIM", "S", "1"):
            carimbo_esperado += 1
            # Localizado = o OCR tirou texto reconhecível DAQUELA
            # prancha (mínimo de palavras plausíveis). Não vale contar
            # campo preenchido, que a propagação pode ter trazido de
            # outra prancha.
            # Exige vocabulário de carimbo de verdade (ESCALA, DATA,
            # ARQUITETO, RUA...). Contar só "palavras" deixava passar
            # lixo de OCR de imagem espelhada ("ATAG", "oalaa"), que
            # é alfabético mas não significa nada.
            if _contar_acertos_aproximados(normalizar(obtido.texto_ocr)) >= ACERTOS_MINIMOS_CARIMBO_LIDO:
                carimbo_encontrado += 1

        for campo, modo in CAMPOS_AVALIADOS.items():
            valor_esperado = normalizar(esperado.get(campo, ""))
            if not valor_esperado:
                continue  # campo não cobrado nesta linha

            placar = placares[campo]
            placar.cobrados += 1
            valor_obtido = normalizar(getattr(obtido, ATRIBUTO[campo], ""))

            if modo == "exato":
                acertou = valor_obtido == valor_esperado
            else:
                acertou = bool(valor_obtido) and (
                    valor_esperado in valor_obtido or valor_obtido in valor_esperado
                )

            if acertou:
                placar.acertos += 1
            else:
                placar.erros += 1
                if len(placar.exemplos_erro) < 3:
                    placar.exemplos_erro.append(
                        f"{nome_arquivo}: esperado={valor_esperado!r} obtido={valor_obtido!r}"
                    )

    return placares, carimbo_esperado, carimbo_encontrado


def imprimir_placar(placares: dict, carimbo_esperado: int, carimbo_encontrado: int, total: int) -> None:
    print()
    print("=" * 64)
    print(f"PLACAR — {total} prancha(s) no gabarito")
    print("=" * 64)

    if carimbo_esperado:
        taxa = carimbo_encontrado / carimbo_esperado
        print(f"{'carimbo lido':<20} {taxa:6.1%}  ({carimbo_encontrado}/{carimbo_esperado})")
        print("-" * 64)

    for campo, placar in placares.items():
        if not placar.cobrados:
            continue
        print(f"{campo:<20} {placar.taxa:6.1%}  ({placar.acertos}/{placar.cobrados})")

    algum_erro = any(p.exemplos_erro for p in placares.values())
    if algum_erro:
        print()
        print("Exemplos de erro (até 3 por campo):")
        for campo, placar in placares.items():
            for exemplo in placar.exemplos_erro:
                print(f"  [{campo}] {exemplo}")
    print()


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Avalia o CAMP Vision contra um gabarito.")
    parser.add_argument("--pasta", required=True, type=Path, help="Pasta com as pranchas de referência")
    parser.add_argument("--gabarito", required=True, type=Path, help="CSV com os valores esperados")
    parser.add_argument("--modelo", default=None, help="Caminho do best.pt (omitir = usar heurística)")
    parser.add_argument("--confianca", default=0.3, type=float, help="Confiança mínima do modelo treinado")
    parser.add_argument("--saida", default=None, type=Path, help="Onde gravar a saída (padrão: pasta temporária)")
    args = parser.parse_args()

    if not args.pasta.is_dir():
        print(f"Pasta não encontrada: {args.pasta}", file=sys.stderr)
        return 1
    if not args.gabarito.is_file():
        print(f"Gabarito não encontrado: {args.gabarito}", file=sys.stderr)
        return 1

    pasta_saida = args.saida or Path(tempfile.mkdtemp(prefix="campvision_bench_"))
    modo = f"modelo treinado ({args.modelo}, conf>={args.confianca})" if args.modelo else "heurística"
    print(f"Avaliando {args.pasta.name} — modo: {modo}")
    print("(os arquivos avaliados NÃO são renomeados nem movidos)")

    import config as app_config
    resultado = processar(
        args.pasta, pasta_saida, args.modelo, args.confianca, app_config.settings.ocr_idiomas,
    )
    gabarito = carregar_gabarito(args.gabarito)
    placares, carimbo_esperado, carimbo_encontrado = comparar(gabarito, resultado)
    imprimir_placar(placares, carimbo_esperado, carimbo_encontrado, len(gabarito))

    print(f"Saída completa em: {pasta_saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
