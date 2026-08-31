#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
automatizado.py
================
Ponto de entrada HEADLESS do CAMP Vision — pensado para rodar sem
ninguém sentado na frente, como serviço systemd num notebook Linux
com acesso ao QNAP e à internet.

Reaproveita 100% do motor já testado (scanner.lote.ProcessadorLote) —
não duplica nenhuma lógica de OCR, IA, classificação ou exportação.
A única diferença real em relação ao app.py (GUI) é:

  1. Não há interface gráfica — o loop principal fica de olho no QNAP.
  2. Renomeação dos arquivos fica DESLIGADA durante a catalogação — os
     arquivos já chegam com o padrão CAMP aplicado pela estação Windows.
  3. Só processa a pasta "03 - Preview (JPG)" de cada projeto. Quando o
     JPG está abaixo do piso de resolução, usa o TIFF correspondente
     apenas para a análise, sem copiar nem alterar o original.
  4. Depois da catalogação, normaliza a estrutura interna do projeto e
     move a pasta inteira para o fundo correto, reaproveitando o fundo
     existente ou criando-o com o arquiteto catalogado quando necessário.

Fluxo:

    status.json == "enviado_windows"
        -> roda o lote (OCR/IA/classificação/EXIF/exportação)
        -> calcula nível de confiança por prancha
        -> normaliza/move o projeto para o fundo correto
        -> status.json = "campvision_concluido"

Uso:
    python automatizado.py                  # roda em loop contínuo
    python automatizado.py --uma-vez       # processa o que houver e sai
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import config as app_config
from ai.interpretador import InterpretadorIA
from classificacao.classificador import criar_classificador
from database.models import criar_engine, criar_sessao
from exportacao.exportador import Exportador
from ocr.motor import criar_motor_ocr
from scanner.detector_carimbo import criar_detector
from scanner.leitor_imagem import RESOLUCAO_MINIMA_PIXELS_SEM_DPI
from scanner.lote import ConfiguracaoLote, ProcessadorLote
from utils.organizacao_final import organizar_projeto_catalogado

# Mesmos nomes de pasta definidos no padrão CAMP (ver sistema_windows.py,
# peça Windows) — precisam ficar em sincronia manual entre os dois
# projetos, já que são bases de código separadas.
SERIE_PADRAO = "01 - Desenhos e Pranchas"
TIPO_PREVIEW = "03 - Preview (JPG)"
TIPO_ARQUIVISTICO = "01 - Arquivo Arquivístico (TIFF)"

ETAPA_ENTRADA = "enviado_windows"
ETAPA_SAIDA = "campvision_concluido"

logger = logging.getLogger("campvision.automatizado")


# ============================================================================
# CONFIGURAÇÃO ESPECÍFICA DO MODO AUTOMATIZADO
# ============================================================================

@dataclass
class ConfigAutomatizado:
    qnap_acervos_path: str = "/mnt/qnap/acervos"
    intervalo_verificacao_segundos: int = 60
    quantidade_threads: int = 2
    timeout_qnap_travado_horas: float = 6.0


def carregar_config_automatizado() -> ConfigAutomatizado:
    caminho = app_config.USER_DIR / "automatizado_config.json"
    app_config.garantir_diretorios()
    if not caminho.exists():
        cfg = ConfigAutomatizado()
        caminho.write_text(
            json.dumps(cfg.__dict__, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.warning(
            "Criado %s com valores padrão — edite qnap_acervos_path antes de usar de verdade.",
            caminho,
        )
        return cfg
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        return ConfigAutomatizado(**{k: v for k, v in dados.items() if k in ConfigAutomatizado.__dataclass_fields__})
    except Exception as exc:
        logger.error("Falha ao ler %s (%s) — usando padrões.", caminho, exc)
        return ConfigAutomatizado()


# ============================================================================
# NÍVEL DE CONFIANÇA
# ============================================================================

PENALIDADE_SEM_IA = 0.7


def _calcular_nivel(confianca_ocr: Optional[float], confianca_ia: Optional[float], observacoes: Optional[str]) -> dict:
    tem_erro = bool(observacoes) and "erro" in observacoes.lower()
    if tem_erro:
        return {"score": 0.0, "nivel": "revisao_manual"}
    if confianca_ocr is None:
        return {"score": 0.0, "nivel": "revisao_manual"}

    score = confianca_ocr
    if confianca_ia is not None and confianca_ia < 0.9:
        score *= PENALIDADE_SEM_IA

    if score >= 0.90:
        nivel = "auto_aprovado"
    elif score >= 0.75:
        nivel = "confirmacao_rapida"
    else:
        nivel = "revisao_manual"

    return {"score": round(score, 3), "nivel": nivel}


def gerar_niveis_confianca(pasta_catalogacao: Path) -> Optional[Path]:
    """Lê o catalogacao.json e produz o arquivo companheiro de níveis."""
    caminho_json = pasta_catalogacao / "catalogacao.json"
    if not caminho_json.exists():
        logger.warning("catalogacao.json não encontrado em %s — pulando cálculo de níveis.", pasta_catalogacao)
        return None

    try:
        registros = json.loads(caminho_json.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Falha ao ler catalogacao.json (%s).", exc)
        return None

    niveis = []
    contagem = {"auto_aprovado": 0, "confirmacao_rapida": 0, "revisao_manual": 0}
    for reg in registros:
        resultado = _calcular_nivel(
            reg.get("Confiança OCR"), reg.get("Confiança IA"), reg.get("Observações")
        )
        contagem[resultado["nivel"]] += 1
        niveis.append({
            "arquivo": reg.get("Arquivo"),
            "projeto": reg.get("Projeto"),
            **resultado,
        })

    saida = {
        "gerado_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "resumo": contagem,
        "pranchas": niveis,
    }
    caminho_saida = pasta_catalogacao / "niveis_confianca.json"
    caminho_saida.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "Níveis de confiança: %d auto-aprovado(s), %d confirmação rápida, %d revisão manual.",
        contagem["auto_aprovado"], contagem["confirmacao_rapida"], contagem["revisao_manual"],
    )
    return caminho_saida


# ============================================================================
# DESCOBERTA DE PROJETOS PENDENTES NO QNAP
# ============================================================================

def _ler_status(pasta_projeto: Path) -> Optional[dict]:
    caminho = pasta_projeto / "status.json"
    if not caminho.exists():
        return None
    try:
        return json.loads(caminho.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("status.json inválido em %s (%s).", pasta_projeto, exc)
        return None


def _escrever_status(pasta_projeto: Path, etapa: str) -> None:
    status = {
        "etapa": etapa,
        "atualizado_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "atualizado_por": "campvision_automatizado",
    }
    (pasta_projeto / "status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def listar_projetos_pendentes(qnap_acervos_path: Path) -> list[Path]:
    """Anda dois níveis (Fundo/ -> Projeto/), igual ao padrão CAMP."""
    pendentes = []
    if not qnap_acervos_path.exists():
        logger.error("Pasta do QNAP não encontrada/montada: %s", qnap_acervos_path)
        return pendentes

    for pasta_fundo in sorted(qnap_acervos_path.iterdir()):
        if not pasta_fundo.is_dir():
            continue
        for pasta_projeto in sorted(pasta_fundo.iterdir()):
            if not pasta_projeto.is_dir():
                continue
            status = _ler_status(pasta_projeto)
            if status and status.get("etapa") == ETAPA_ENTRADA:
                pendentes.append(pasta_projeto)
    return pendentes


# ============================================================================
# PROCESSAMENTO DE UM PROJETO
# ============================================================================

def _preparar_pasta_processamento(pasta_projeto: Path, pasta_jpg: Path) -> tuple[Path, list[str]]:
    """Monta uma pasta temporária de links simbólicos para análise."""
    pasta_tif = pasta_projeto / SERIE_PADRAO / TIPO_ARQUIVISTICO
    pasta_temp = Path(tempfile.mkdtemp(prefix="campvision_prep_"))
    avisos = []

    for jpg in sorted(pasta_jpg.glob("*.jpg")) + sorted(pasta_jpg.glob("*.jpeg")):
        origem = jpg
        motivo = ""
        try:
            from PIL import Image
            with Image.open(jpg) as img:
                largura, altura = img.size
            if min(largura, altura) < RESOLUCAO_MINIMA_PIXELS_SEM_DPI:
                candidato_tif = None
                for ext in (".tif", ".tiff"):
                    possivel = pasta_tif / (jpg.stem + ext)
                    if possivel.exists():
                        candidato_tif = possivel
                        break
                if candidato_tif is not None:
                    origem = candidato_tif
                    motivo = (
                        f"{jpg.name}: JPG com {largura}x{altura}px (abaixo do piso de "
                        f"{RESOLUCAO_MINIMA_PIXELS_SEM_DPI}px) — usando o TIFF original "
                        f"({candidato_tif.name}) só para esta análise."
                    )
                else:
                    motivo = (
                        f"{jpg.name}: JPG com {largura}x{altura}px (abaixo do piso de "
                        f"{RESOLUCAO_MINIMA_PIXELS_SEM_DPI}px), mas não achei o TIFF "
                        f"correspondente em '{TIPO_ARQUIVISTICO}' — seguindo com o JPG mesmo."
                    )
        except Exception as exc:
            motivo = f"{jpg.name}: não consegui checar a resolução ({exc}) — seguindo com o JPG."

        if motivo:
            avisos.append(motivo)

        destino = pasta_temp / (jpg.stem + origem.suffix.lower())
        os.symlink(origem, destino)

    return pasta_temp, avisos


def processar_projeto(pasta_projeto: Path, cfg: ConfigAutomatizado, settings: app_config.Settings) -> bool:
    """Cataloga e, ao final, organiza o projeto no fundo correto."""
    pasta_jpg = pasta_projeto / SERIE_PADRAO / TIPO_PREVIEW
    if not pasta_jpg.exists():
        logger.error("Pasta de preview não encontrada em %s — pulando.", pasta_projeto)
        return False

    pasta_catalogacao = pasta_projeto / "catalogacao"
    pasta_catalogacao.mkdir(parents=True, exist_ok=True)
    rotulo_projeto = f"{pasta_projeto.parent.name} / {pasta_projeto.name}"

    pasta_temp, avisos_resolucao = _preparar_pasta_processamento(pasta_projeto, pasta_jpg)
    for aviso in avisos_resolucao:
        logger.info("[resolução] %s", aviso)

    try:
        config_lote = ConfiguracaoLote(
            pasta_entrada=pasta_temp,
            pasta_saida=pasta_catalogacao,
            formatos_aceitos=[".jpg", ".jpeg", ".tif", ".tiff"],
            quantidade_threads=cfg.quantidade_threads,
            idiomas_ocr=settings.ocr_idiomas,
            tamanho_miniatura=settings.miniatura_tamanho_px,
            qualidade_miniatura=settings.miniatura_qualidade,
            salvar_miniaturas=settings.salvar_miniaturas,
            salvar_carimbos=settings.salvar_carimbos,
            ia_api_key=settings.ia_api_key,
            ia_modelo=settings.ia_modelo,
            ia_habilitada=settings.ia_habilitada,
            caminho_db=str(app_config.DB_PATH),
            ocr_motor=settings.ocr_motor,
            deteccao_carimbo_modo=settings.deteccao_carimbo_modo,
            caminho_modelo_carimbo=settings.caminho_modelo_carimbo,
            carimbo_regiao_busca=settings.carimbo_regiao_busca,
            classificacao_modo=settings.classificacao_modo,
            caminho_modelo_classificacao=settings.caminho_modelo_classificacao,
            confianca_minima_ml=settings.confianca_minima_ml,
            tamanho_imagem_ml=settings.tamanho_imagem_ml,
            codigo_projeto=rotulo_projeto,
            renomeacao_habilitada=False,
            arquivamento_habilitado=False,
            gravar_metadados_exif=settings.gravar_metadados_exif,
            deteccao_multiorientacao=settings.deteccao_multiorientacao,
            nome_projeto=pasta_projeto.name,
            atribuicao_instituicao=settings.atribuicao_instituicao,
        )

        logger.info("=== Processando projeto: %s ===", rotulo_projeto)
        try:
            processador = ProcessadorLote(config_lote)
            resultados = processador.executar()
        except Exception as exc:
            logger.exception("Falha ao processar projeto %s: %s", rotulo_projeto, exc)
            return False
    finally:
        shutil.rmtree(pasta_temp, ignore_errors=True)

    sucesso = sum(1 for r in resultados if r.sucesso)
    falhas = sum(1 for r in resultados if not r.sucesso)
    logger.info("Projeto %s: %d prancha(s) catalogada(s), %d falha(s).", rotulo_projeto, sucesso, falhas)

    gerar_niveis_confianca(pasta_catalogacao)

    # Só considera o projeto concluído depois de colocá-lo na estrutura
    # definitiva. Se a organização falhar, ele continua com
    # 'enviado_windows' e será tentado novamente na próxima rodada.
    destino_final = organizar_projeto_catalogado(
        pasta_projeto,
        Path(cfg.qnap_acervos_path),
        db_path=str(app_config.DB_PATH),
    )
    if destino_final is None:
        logger.warning(
            "Projeto %s foi catalogado, mas NÃO foi organizado no arquivo definitivo; "
            "continua 'enviado_windows' para nova tentativa.",
            pasta_projeto.name,
        )
        return False

    _escrever_status(destino_final, ETAPA_SAIDA)
    return True


# ============================================================================
# LOOP PRINCIPAL
# ============================================================================

ARQUIVO_TRAVA = app_config.USER_DIR / "processando.lock"


def executar_uma_rodada(cfg: ConfigAutomatizado, settings: app_config.Settings) -> int:
    qnap = Path(cfg.qnap_acervos_path)
    pendentes = listar_projetos_pendentes(qnap)
    if not pendentes:
        logger.info("Nada pendente no momento.")
        return 0

    logger.info("%d projeto(s) pendente(s) encontrado(s).", len(pendentes))
    processados = 0
    for pasta_projeto in pendentes:
        ARQUIVO_TRAVA.write_text(str(time.time()), encoding="utf-8")
        try:
            ok = processar_projeto(pasta_projeto, cfg, settings)
            if ok:
                processados += 1
            else:
                logger.warning(
                    "Projeto %s NÃO avançou de etapa — continua 'enviado_windows', "
                    "será tentado de novo na próxima rodada.",
                    pasta_projeto.name,
                )
        except Exception as exc:
            logger.exception("Erro inesperado processando %s: %s", pasta_projeto, exc)
        finally:
            ARQUIVO_TRAVA.unlink(missing_ok=True)
    return processados


def main() -> int:
    parser = argparse.ArgumentParser(description="CAMP Vision — modo automatizado (headless)")
    parser.add_argument("--uma-vez", action="store_true", help="Processa o que houver pendente e sai (sem loop).")
    args = parser.parse_args()

    app_config.garantir_diretorios()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(app_config.LOG_DIR / "automatizado.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    criar_engine(str(app_config.DB_PATH))
    settings = app_config.carregar_configuracoes()
    cfg = carregar_config_automatizado()

    logger.info("=" * 60)
    logger.info("CAMP Vision — modo automatizado — %s", app_config.VERSAO_BUILD)
    logger.info("QNAP: %s | intervalo: %ds", cfg.qnap_acervos_path, cfg.intervalo_verificacao_segundos)
    logger.info("=" * 60)

    if args.uma_vez:
        executar_uma_rodada(cfg, settings)
        return 0

    while True:
        try:
            executar_uma_rodada(cfg, settings)
        except Exception as exc:
            logger.exception("Erro na rodada de verificação: %s", exc)
        time.sleep(cfg.intervalo_verificacao_segundos)


if __name__ == "__main__":
    sys.exit(main())
