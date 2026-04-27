# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for Spike Doctor.

Usage:
    pyinstaller SpikeDoctor.spec

This bundles the Shiny web application into a standalone Windows executable.
Build must be performed ON Windows (PyInstaller cannot cross-compile).
"""

from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None

# Collect all submodules, data files, and binaries for packages that
# PyInstaller cannot auto-detect.
shiny_datas, shiny_binaries, shiny_hidden = collect_all("shiny")
efel_datas, efel_binaries, efel_hidden = collect_all("efel")
matplotlib_datas = collect_data_files("matplotlib")

a = Analysis(
    ["pyinstaller_entry.py"],
    pathex=[],
    binaries=shiny_binaries + efel_binaries,
    datas=[
        ("app.py", "."),
        ("modules", "modules"),
        ("assets", "assets"),
    ] + shiny_datas + efel_datas + matplotlib_datas,
    hiddenimports=shiny_hidden + efel_hidden + [
        "matplotlib",
        "matplotlib.pyplot",
        "matplotlib.backends.backend_agg",
        "matplotlib.backends.backend_pdf",
    ],
    hookspath=[],
    hooksconfig={
        "matplotlib": {
            "backends": ["Agg"],
        },
    },
    runtime_hooks=[],
    excludes=[
        "matplotlib.tests",
        "chatlas",
        "anthropic",
        "openai",
        "google.generativeai",
        "google_genai",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="SpikeDoctor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon="assets/spike-doctor-icon.ico",  # Uncomment if icon exists
)
