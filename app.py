"""
app.py
======
Ponto de entrada do CAMP Vision.

Inicializa o banco de dados e exibe a janela principal (Tkinter).
"""

from __future__ import annotations

import os
import sys

# Silencia o aviso "OMP: Warning #191: Forking a process while a
# parallel region is active" — ele aparece dezenas de vezes por lote
# e torna o log ilegível. A causa é legítima e inofensiva aqui: o
# Tesseract é executado como subprocesso enquanto o runtime OpenMP
# (usado pelo torch/OpenCV) mantém uma região paralela ativa. Precisa
# ser definido ANTES de qualquer import que carregue OpenMP.
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import config as app_config
from database.models import criar_engine
from interface.janela_principal import JanelaPrincipal


def main() -> int:
    app_config.garantir_diretorios()
    criar_engine(str(app_config.DB_PATH))  # garante schema do banco de conhecimento

    print("=" * 50)
    print(f"  {app_config.APP_NAME} — build {app_config.VERSAO_BUILD}")
    print("=" * 50)

    janela = JanelaPrincipal()
    janela.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
