"""
interface/temas.py
===================
Aplica uma paleta de cores clara ou escura à interface Tkinter/ttk.

Tkinter não tem um sistema de dark mode tão completo quanto o Qt,
mas no macOS (com um Python/Tcl-Tk relativamente recente) a janela já
tende a seguir a aparência do sistema automaticamente. As paletas
abaixo garantem contraste adequado nos dois casos e funcionam mesmo
em versões mais antigas do Tcl/Tk (comuns em instalações de Python
mais antigas, como as compatíveis com Macs mais velhos).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

PALETA_CLARA = {
    "fundo": "#f5f5f7",
    "fundo_widget": "#ffffff",
    "texto": "#1c1c1e",
    "destaque": "#007aff",
    "destaque_texto": "#ffffff",
    "borda": "#c7c7cc",
    "sucesso": "#34c759",
}

PALETA_ESCURA = {
    "fundo": "#1e1e1e",
    "fundo_widget": "#2c2c2e",
    "texto": "#e8e8ea",
    "destaque": "#0a84ff",
    "destaque_texto": "#ffffff",
    "borda": "#3a3a3c",
    "sucesso": "#30d158",
}


def obter_paleta(nome_tema: str) -> dict:
    return PALETA_ESCURA if nome_tema == "escuro" else PALETA_CLARA


def aplicar_tema(root: tk.Tk, nome_tema: str) -> dict:
    """Aplica a paleta de cores à janela raiz e configura os estilos
    ttk usados pelos widgets do CAMP Vision. Retorna a paleta, para
    que a janela possa reutilizar as cores em widgets Tk "clássicos"
    (não-ttk), como o ScrolledText do log."""
    paleta = obter_paleta(nome_tema)

    root.configure(bg=paleta["fundo"])

    estilo = ttk.Style(root)
    try:
        estilo.theme_use("clam")  # tema mais consistente entre plataformas/versões do Tcl-Tk
    except tk.TclError:
        pass  # se 'clam' não estiver disponível, mantém o tema padrão do sistema

    estilo.configure("TFrame", background=paleta["fundo"])
    estilo.configure("TLabelframe", background=paleta["fundo"], foreground=paleta["texto"])
    estilo.configure("TLabelframe.Label", background=paleta["fundo"], foreground=paleta["texto"])
    estilo.configure("TLabel", background=paleta["fundo"], foreground=paleta["texto"])
    estilo.configure(
        "TButton", background=paleta["destaque"], foreground=paleta["destaque_texto"],
        borderwidth=0, focuscolor=paleta["destaque"], padding=6,
    )
    estilo.map("TButton", background=[("disabled", paleta["borda"]), ("active", paleta["destaque"])])
    estilo.configure(
        "TEntry", fieldbackground=paleta["fundo_widget"], foreground=paleta["texto"],
        insertcolor=paleta["texto"],
    )
    estilo.configure(
        "Horizontal.TProgressbar", background=paleta["sucesso"], troughcolor=paleta["fundo_widget"],
    )
    estilo.configure(
        "TCombobox", fieldbackground=paleta["fundo_widget"], foreground=paleta["texto"],
    )
    estilo.configure("TCheckbutton", background=paleta["fundo"], foreground=paleta["texto"])
    estilo.configure("TSpinbox", fieldbackground=paleta["fundo_widget"], foreground=paleta["texto"])

    return paleta
