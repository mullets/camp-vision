"""
ml/classificador_ml.py
========================
Inferência do classificador de tipo de prancha treinado (ResNet18
fine-tuned via `ml/treinar_classificador.py`), expondo o mesmo tipo
de resultado usado pelo classificador por regras
(`classificacao.classificador.ResultadoClassificacao`).
"""

from __future__ import annotations

import logging

import numpy as np

from classificacao.classificador import ResultadoClassificacao

logger = logging.getLogger("campvision.ml.classificador")

TAMANHO_ENTRADA = 224


class ClassificadorML:
    """Wrapper de inferência do modelo treinado de classificação de
    tipo de prancha. Carrega os pesos uma única vez por instância."""

    def __init__(self, caminho_modelo: str, dispositivo: str | None = None):
        import torch
        from torch import nn
        from torchvision import models

        checkpoint = torch.load(caminho_modelo, map_location="cpu")
        self.classes: list[str] = checkpoint["classes"]

        self.dispositivo = dispositivo or (
            "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        modelo = models.resnet18(weights=None)
        modelo.fc = nn.Linear(modelo.fc.in_features, len(self.classes))
        modelo.load_state_dict(checkpoint["state_dict"])
        modelo.to(self.dispositivo)
        modelo.eval()

        self._modelo = modelo
        self._torch = torch
        self._transform = self._construir_transform()

        logger.info("Modelo de classificação carregado (%d classes, dispositivo=%s)",
                     len(self.classes), self.dispositivo)

    def _construir_transform(self):
        from torchvision import transforms
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((TAMANHO_ENTRADA, TAMANHO_ENTRADA)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def classificar(self, imagem_bgr: np.ndarray) -> ResultadoClassificacao:
        """Classifica a imagem completa da prancha (não o carimbo),
        retornando a categoria mais provável e a confiança do modelo."""
        import cv2

        imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(imagem_rgb).unsqueeze(0).to(self.dispositivo)

        with self._torch.no_grad():
            saida = self._modelo(tensor)
            probabilidades = self._torch.softmax(saida, dim=1)[0]

        indice = int(probabilidades.argmax())
        confianca = float(probabilidades[indice])
        tipo = self.classes[indice]

        return ResultadoClassificacao(tipo=tipo, confianca=round(confianca, 3), evidencias=["modelo treinado"])
