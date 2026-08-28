"""
utils/tempo.py
===============
Cálculo de tempo decorrido e estimativa de tempo restante (ETA)
durante o processamento em lote, usado pela barra de progresso da
interface gráfica.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class EstimadorTempo:
    """Estima o tempo restante com base na média móvel do tempo por item."""

    total_itens: int
    inicio: float = field(default_factory=time.monotonic)
    itens_concluidos: int = 0

    def registrar_item_concluido(self) -> None:
        self.itens_concluidos += 1

    @property
    def tempo_decorrido_segundos(self) -> float:
        return time.monotonic() - self.inicio

    @property
    def tempo_medio_por_item(self) -> float:
        if self.itens_concluidos == 0:
            return 0.0
        return self.tempo_decorrido_segundos / self.itens_concluidos

    @property
    def tempo_restante_segundos(self) -> float:
        restantes = max(0, self.total_itens - self.itens_concluidos)
        return restantes * self.tempo_medio_por_item

    def formatar_duracao(self, segundos: float) -> str:
        segundos = int(segundos)
        horas, resto = divmod(segundos, 3600)
        minutos, seg = divmod(resto, 60)
        if horas:
            return f"{horas}h {minutos}m {seg}s"
        if minutos:
            return f"{minutos}m {seg}s"
        return f"{seg}s"

    @property
    def eta_formatado(self) -> str:
        return self.formatar_duracao(self.tempo_restante_segundos)
