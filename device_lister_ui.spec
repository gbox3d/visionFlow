# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs


project_root = Path(globals().get("SPECPATH", ".")).resolve()

binaries = []
binaries += collect_dynamic_libs("cv2")

a = Analysis(
    ["device_lister_ui.py"],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=[],
    hiddenimports=[
        "cv2_enumerate_cameras",
        "sounddevice",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="device_lister_ui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="device_lister_ui",
)

