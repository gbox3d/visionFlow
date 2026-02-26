# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

project_root = Path(globals().get("SPECPATH", ".")).resolve()
src_root = project_root / "src"

datas = []
if (project_root / ".env").exists():
    datas.append((str(project_root / ".env"), "."))

hiddenimports = [
    "cv2_enumerate_cameras",
    "sounddevice",
    "dotenv",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
]

a = Analysis(
    ["src/voiceFlow/sample/mic_level_monitor.py"],
    pathex=[str(project_root), str(src_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "cv2",
        "torch",
        "torchaudio",
        "torchvision",
        "faster_whisper",
        "ctranslate2",
        "transformers",
        "accelerate",
        "mediapipe",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="mic_level_monitor",
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
    name="mic_level_monitor",
)
