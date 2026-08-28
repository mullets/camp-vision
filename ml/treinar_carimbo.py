"""
ml/treinar_carimbo.py
======================
Treina um modelo YOLO (Ultralytics) para localizar o carimbo em
pranchas arquitetônicas, substituindo (ou complementando, como
verificação cruzada) a heurística de contornos em
`scanner/detector_carimbo.py`.

Uso via linha de comando:

    python -m ml.treinar_carimbo \\
        --anotacoes /caminho/para/anotacoes \\
        --saida /caminho/para/dataset_preparado \\
        --modelos-saida modelos/ \\
        --epocas 150 \\
        --tamanho-imagem 960

O modelo final é salvo em `<modelos-saida>/treino_carimbo/weights/best.pt`,
que deve ser apontado em `config.settings.caminho_modelo_carimbo`.

Requer o pacote `ultralytics` (ver requirements.txt). O treinamento é
significativamente mais rápido em GPU (CUDA ou Apple Silicon MPS),
mas funciona em CPU para datasets pequenos/testes.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ml.dataset_carimbo import preparar_dataset

logger = logging.getLogger("campvision.ml.treinar_carimbo")

MODELO_BASE_PADRAO = "yolov8n.pt"  # variante "nano": leve, ideal para uma única classe


def treinar(
    dataset_yaml: Path,
    modelos_saida: Path,
    modelo_base: str = MODELO_BASE_PADRAO,
    epocas: int = 150,
    tamanho_imagem: int = 960,
    lote: int = 8,
):
    """Executa o fine-tuning do YOLO sobre o dataset de carimbos."""
    from ultralytics import YOLO  # import tardio: dependência pesada e opcional

    logger.info("Iniciando treinamento a partir de %s (%d épocas)", modelo_base, epocas)
    modelo = YOLO(modelo_base)
    resultado = modelo.train(
        data=str(dataset_yaml),
        epochs=epocas,
        imgsz=tamanho_imagem,
        batch=lote,
        project=str(modelos_saida),
        name="treino_carimbo",
        patience=20,          # early stopping se não houver melhora
        exist_ok=True,
    )
    logger.info("Treinamento concluído. Pesos salvos em %s", modelos_saida / "treino_carimbo" / "weights" / "best.pt")
    return resultado


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Treina o detector de carimbo (YOLO) do CAMP Vision.")
    parser.add_argument("--anotacoes", required=True, type=Path,
                         help="Pasta com pares imagem+anotação .txt (formato YOLO)")
    parser.add_argument("--saida", required=True, type=Path,
                         help="Pasta onde o dataset organizado (train/val) será criado")
    parser.add_argument("--modelos-saida", default=Path("modelos"), type=Path,
                         help="Pasta onde os pesos treinados serão salvos")
    parser.add_argument("--modelo-base", default=MODELO_BASE_PADRAO,
                         help="Checkpoint YOLO base para fine-tuning (padrão: yolov8n.pt)")
    parser.add_argument("--epocas", default=150, type=int)
    parser.add_argument("--tamanho-imagem", default=960, type=int)
    parser.add_argument("--lote", default=8, type=int)
    parser.add_argument("--proporcao-validacao", default=0.15, type=float)

    args = parser.parse_args()

    resumo = preparar_dataset(
        pasta_anotacoes=args.anotacoes,
        pasta_saida=args.saida,
        proporcao_validacao=args.proporcao_validacao,
    )
    logger.info("Dataset: %d treino / %d validação", resumo.total_treino, resumo.total_validacao)

    treinar(
        dataset_yaml=resumo.caminho_yaml,
        modelos_saida=args.modelos_saida,
        modelo_base=args.modelo_base,
        epocas=args.epocas,
        tamanho_imagem=args.tamanho_imagem,
        lote=args.lote,
    )


if __name__ == "__main__":
    main()
