# -*- mode: python ; coding: utf-8 -*-
"""
Especificação do PyInstaller para empacotar o CAMP Vision como um
aplicativo nativo (.app) para macOS.

Usa Tkinter (biblioteca padrão) em vez de Qt/PySide6 — gera um
aplicativo bem mais leve e sem as exigências de macOS recente que
frameworks Qt costumam ter, importante para Macs mais antigos.

Uso:
    pyinstaller campvision.spec
"""

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ["app.py"],
    pathex=[str(Path.cwd())],
    binaries=[],
    datas=[],
    hiddenimports=[
        "tkinter",
        "openai",
        "sqlalchemy",
        "pytesseract",
        "openpyxl",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CAMP Vision",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="CAMP Vision",
)

app = BUNDLE(
    coll,
    name="CAMP Vision.app",
    icon=None,  # aponte para "recursos/icone.icns" quando disponível
    bundle_identifier="com.campvision.app",
    info_plist={
        "NSHighResolutionCapable": "True",
        "CFBundleShortVersionString": "0.1.0",
        "NSHumanReadableCopyright": "CAMP Vision",
        "LSMinimumSystemVersion": "10.13",
    },
)
