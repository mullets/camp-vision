"""
ml/treinar_classificador.py
=============================
Treina um classificador de tipo de prancha por imagem, via
fine-tuning de uma ResNet18 pré-treinada (torchvision), substituindo
(ou complementando) o classificador baseado em palavras-chave em
`classificacao/classificador.py`.

Uso via linha de comando:

    python -m ml.treinar_classificador \\
        --dataset /caminho/para/pasta_dataset \\
        --saida modelos/classificador_tipo.pt \\
        --epocas 20

O dataset deve seguir o formato descrito em
`ml/dataset_classificacao.py` (uma subpasta por categoria).

O modelo final salvo em `--saida` deve ser apontado em
`config.settings.caminho_modelo_classificacao`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from classificacao.classificador import CATEGORIAS
from ml.dataset_classificacao import validar_dataset

logger = logging.getLogger("campvision.ml.treinar_classificador")

TAMANHO_ENTRADA = 224


def _construir_transforms():
    from torchvision import transforms

    transform_treino = transforms.Compose([
        transforms.Resize((TAMANHO_ENTRADA, TAMANHO_ENTRADA)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_validacao = transforms.Compose([
        transforms.Resize((TAMANHO_ENTRADA, TAMANHO_ENTRADA)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform_treino, transform_validacao


def treinar(
    pasta_dataset: Path,
    caminho_saida: Path,
    epocas: int = 20,
    taxa_aprendizado: float = 1e-4,
    tamanho_lote: int = 16,
    proporcao_validacao: float = 0.15,
) -> None:
    """Executa o fine-tuning de uma ResNet18 sobre o dataset de tipos
    de prancha e salva os pesos treinados em `caminho_saida`."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, models

    resumo = validar_dataset(pasta_dataset)
    if not resumo.valido():
        raise ValueError(
            f"Dataset inválido: classes ausentes = {resumo.classes_ausentes}. "
            "Crie uma subpasta por categoria com pelo menos alguns exemplos."
        )

    transform_treino, transform_validacao = _construir_transforms()

    dataset_completo = datasets.ImageFolder(str(pasta_dataset), transform=transform_treino)
    classes_dataset = dataset_completo.classes  # ordem alfabética das subpastas encontradas

    tamanho_val = max(1, int(len(dataset_completo) * proporcao_validacao))
    tamanho_treino = len(dataset_completo) - tamanho_val
    dataset_treino, dataset_val = random_split(dataset_completo, [tamanho_treino, tamanho_val])
    dataset_val.dataset.transform = transform_validacao  # val não usa augmentation

    loader_treino = DataLoader(dataset_treino, batch_size=tamanho_lote, shuffle=True, num_workers=2)
    loader_val = DataLoader(dataset_val, batch_size=tamanho_lote, shuffle=False, num_workers=2)

    dispositivo = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Treinando em dispositivo: %s | classes: %s", dispositivo, classes_dataset)

    modelo = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    modelo.fc = nn.Linear(modelo.fc.in_features, len(classes_dataset))
    modelo.to(dispositivo)

    otimizador = torch.optim.Adam(modelo.parameters(), lr=taxa_aprendizado)
    funcao_perda = nn.CrossEntropyLoss()

    melhor_acuracia_val = 0.0

    for epoca in range(1, epocas + 1):
        modelo.train()
        perda_acumulada = 0.0
        for entradas, rotulos in loader_treino:
            entradas, rotulos = entradas.to(dispositivo), rotulos.to(dispositivo)
            otimizador.zero_grad()
            saidas = modelo(entradas)
            perda = funcao_perda(saidas, rotulos)
            perda.backward()
            otimizador.step()
            perda_acumulada += perda.item() * entradas.size(0)

        perda_media = perda_acumulada / len(dataset_treino)
        acuracia_val = _avaliar(modelo, loader_val, dispositivo)
        logger.info("Época %d/%d — perda treino: %.4f — acurácia val: %.2f%%",
                     epoca, epocas, perda_media, acuracia_val * 100)

        if acuracia_val >= melhor_acuracia_val:
            melhor_acuracia_val = acuracia_val
            caminho_saida.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": modelo.state_dict(), "classes": classes_dataset}, caminho_saida)
            logger.info("Novo melhor modelo salvo (acurácia val: %.2f%%) em %s", acuracia_val * 100, caminho_saida)

    logger.info("Treinamento concluído. Melhor acurácia de validação: %.2f%%", melhor_acuracia_val * 100)


def _avaliar(modelo, loader_val, dispositivo) -> float:
    import torch

    modelo.eval()
    acertos, total = 0, 0
    with torch.no_grad():
        for entradas, rotulos in loader_val:
            entradas, rotulos = entradas.to(dispositivo), rotulos.to(dispositivo)
            saidas = modelo(entradas)
            preditos = saidas.argmax(dim=1)
            acertos += (preditos == rotulos).sum().item()
            total += rotulos.size(0)
    return acertos / total if total else 0.0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Treina o classificador de tipo de prancha do CAMP Vision.")
    parser.add_argument("--dataset", required=True, type=Path,
                         help="Pasta com uma subpasta por categoria (ver ml/dataset_classificacao.py)")
    parser.add_argument("--saida", required=True, type=Path,
                         help="Caminho do arquivo .pt onde o modelo treinado será salvo")
    parser.add_argument("--epocas", default=20, type=int)
    parser.add_argument("--taxa-aprendizado", default=1e-4, type=float)
    parser.add_argument("--tamanho-lote", default=16, type=int)

    args = parser.parse_args()

    treinar(
        pasta_dataset=args.dataset,
        caminho_saida=args.saida,
        epocas=args.epocas,
        taxa_aprendizado=args.taxa_aprendizado,
        tamanho_lote=args.tamanho_lote,
    )


if __name__ == "__main__":
    main()
