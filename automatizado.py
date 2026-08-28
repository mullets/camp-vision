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
  2. Renomeação e arquivamento em pasta (ano/projeto) ficam DESLIGADOS
     — os arquivos já chegam renomeados e organizados no padrão CAMP
     (Fundo/Projeto/Série/Tipo) feito pela estação Windows; o CAMP
     Vision aqui só CATALOGA (OCR + IA + classificação + EXIF +
     exportação), sem tocar em nome nem em pasta.
  3. Só processa a pasta "03 - Preview (JPG)" de cada projeto — os
     TIFFs de preservação nunca são tocados por este processo.

Fluxo (reaproveita o semáforo já usado entre Windows/QNAP/Mac):

    status.json == "enviado_windows"
        -> roda o lote (OCR/IA/classificação/EXIF/exportação)
        -> calcula nível de confiança por prancha (ver _calcular_nivel)
        -> status.json = "campvision_concluido"

Uso:
    python automatizado.py                  # roda em loop contínuo
    python automatizado.py --uma-vez         # processa o que houver e sai
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
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
from scanner.lote import ConfiguracaoLote, ProcessadorLote

# Mesmos nomes de pasta definidos no padrão CAMP (ver sistema_windows.py,
# peça Windows) — precisam ficar em sincronia manual entre os dois
# projetos, já que são bases de código separadas.
SERIE_PADRAO = "01 - Desenhos e Pranchas"
TIPO_PREVIEW = "03 - Preview (JPG)"

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
# NÍVEL DE CONFIANÇA (adaptação honesta do esquema de 4 níveis para o
# que realmente temos disponível sem custo extra: confiança média de
# OCR por palavra — real e contínua — penalizada quando a IA não pôde
# ser usada e o sistema caiu para o extrator por regras)
# ============================================================================

PENALIDADE_SEM_IA = 0.7  # multiplicador aplicado quando confianca_ia == 0.4 (fallback)


def _calcular_nivel(confianca_ocr: Optional[float], confianca_ia: Optional[float], observacoes: Optional[str]) -> dict:
    tem_erro = bool(observacoes) and "erro" in observacoes.lower()
    if tem_erro:
        return {"score": 0.0, "nivel": "revisao_manual"}
    if confianca_ocr is None:
        return {"score": 0.0, "nivel": "revisao_manual"}

    score = confianca_ocr
    if confianca_ia is not None and confianca_ia < 0.9:  # 0.4 = usou fallback de regras, não IA
        score *= PENALIDADE_SEM_IA

    if score >= 0.90:
        nivel = "auto_aprovado"
    elif score >= 0.75:
        nivel = "confirmacao_rapida"
    else:
        nivel = "revisao_manual"

    return {"score": round(score, 3), "nivel": nivel}


def gerar_niveis_confianca(pasta_catalogacao: Path) -> Optional[Path]:
    """Lê o catalogacao.json que o Exportador já escreveu e produz um
    arquivo COMPANHEIRO (niveis_confianca.json) com o nível por
    prancha — não modifica o catalogacao.json original do CAMP Vision.

    IMPORTANTE: o catalogacao.json usa as MESMAS chaves em português
    (Title Case) do CSV/XLSX — ver exportacao/exportador.py:
    RegistroExportacao.para_linha_csv() — não os nomes internos do
    dataclass (que são em minúsculo/snake_case)."""
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
# DESCOBERTA DE PROJETOS PENDENTES NO QNAP (mesma estrutura do padrão CAMP)
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
    """Anda dois níveis (Fundo/ -> Projeto/), igual ao que a peça
    Windows já faz — reaproveita a mesma leitura de disco, sem precisar
    de nenhuma API entre os dois lados."""
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

def processar_projeto(pasta_projeto: Path, cfg: ConfigAutomatizado, settings: app_config.Settings) -> bool:
    """Roda o motor já testado do CAMP Vision sobre a pasta de PREVIEW
    (JPG) de um único projeto. Retorna True se processou com sucesso
    (mesmo que algumas pranchas individuais tenham falhado — falha
    parcial não é falha do projeto inteiro)."""
    pasta_jpg = pasta_projeto / SERIE_PADRAO / TIPO_PREVIEW
    if not pasta_jpg.exists():
        logger.error("Pasta de preview não encontrada em %s — pulando.", pasta_projeto)
        return False

    pasta_catalogacao = pasta_projeto / "catalogacao"
    pasta_catalogacao.mkdir(parents=True, exist_ok=True)

    # codigo_projeto aqui é só um rótulo pro CSV/JSON (renomeação está
    # desligada, então não afeta nome nem pasta de arquivo nenhum) —
    # usa o nome da pasta do fundo+projeto pra ficar identificável.
    rotulo_projeto = f"{pasta_projeto.parent.name} / {pasta_projeto.name}"

    config_lote = ConfiguracaoLote(
        pasta_entrada=pasta_jpg,
        pasta_saida=pasta_catalogacao,
        formatos_aceitos=[".jpg", ".jpeg"],
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
        # --- as duas flags que importam pra não brigar com o padrão CAMP ---
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

    sucesso = sum(1 for r in resultados if r.sucesso)
    falhas = sum(1 for r in resultados if not r.sucesso)
    logger.info("Projeto %s: %d prancha(s) catalogada(s), %d falha(s).", rotulo_projeto, sucesso, falhas)

    gerar_niveis_confianca(pasta_catalogacao)
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
        # A trava existe SÓ enquanto um projeto está sendo processado —
        # é o que o script de deploy confere antes de reiniciar o
        # serviço, pra nunca cortar um lote pela metade. Fica fora do
        # try/finally seguinte de propósito: mesmo que dê erro, o
        # finally sempre remove a trava.
        ARQUIVO_TRAVA.write_text(str(time.time()), encoding="utf-8")
        try:
            ok = processar_projeto(pasta_projeto, cfg, settings)
            if ok:
                _escrever_status(pasta_projeto, ETAPA_SAIDA)
                processados += 1
            else:
                logger.warning(
                    "Projeto %s NÃO avançou de etapa — continua 'enviado_windows', "
                    "será tentado de novo na próxima rodada.",
                    pasta_projeto.name,
                )
        except Exception as exc:
            # Resiliência: um projeto com problema nunca pode travar os
            # outros pendentes na mesma rodada.
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
            # Nunca deixa o loop principal morrer por causa de um erro
            # de uma rodada — o systemd também reinicia se isso escapar,
            # mas é melhor logar e tentar de novo sozinho primeiro.
            logger.exception("Erro na rodada de verificação: %s", exc)
        time.sleep(cfg.intervalo_verificacao_segundos)


if __name__ == "__main__":
    sys.exit(main())
