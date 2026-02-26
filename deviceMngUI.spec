# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.building.datastruct import TOC
from PyInstaller.utils.hooks import collect_dynamic_libs


project_root = Path(globals().get("SPECPATH", ".")).resolve()
src_root = project_root / "src"

datas = []
if (project_root / ".env").exists():
    datas.append((str(project_root / ".env"), "."))

binaries = []
binaries += collect_dynamic_libs("cv2")

hiddenimports = [
    "cv2_enumerate_cameras",
    "sounddevice",
    "dotenv",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
]

# Keep the Widgets-only runtime lean:
# - Drop Qt Quick/QML/PDF/SVG stacks that are not used by this tool.
# - Drop OpenCV FFmpeg runtime used for video-file codecs (camera index capture does not require it).
_DROP_BINARY_HINTS = {
    "opencv_videoio_ffmpeg",
    "pyside6/opengl32sw.dll",
    "pyside6/qt6opengl.dll",
    "pyside6/qt6quick.dll",
    "pyside6/qt6qml.dll",
    "pyside6/qt6qmlmodels.dll",
    "pyside6/qt6qmlmeta.dll",
    "pyside6/qt6qmlworkerscript.dll",
    "pyside6/qt6virtualkeyboard.dll",
    "pyside6/qt6pdf.dll",
    "pyside6/qt6svg.dll",
    "pyside6/plugins/platforminputcontexts/qtvirtualkeyboardplugin.dll",
    "pyside6/plugins/imageformats/qpdf.dll",
    "pyside6/plugins/imageformats/qsvg.dll",
}


def _should_drop(entry) -> bool:
    dst = str(entry[0]).replace("\\", "/").lower()
    src = str(entry[1]).replace("\\", "/").lower() if len(entry) > 1 else ""
    probe = f"{dst}|{src}"
    return any(hint in probe for hint in _DROP_BINARY_HINTS)

a = Analysis(
    ["deviceMngUI.py"],
    pathex=[str(project_root), str(src_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
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
a.binaries = TOC([entry for entry in a.binaries if not _should_drop(entry)])
if getattr(a, "datas", None):
    a.datas = TOC([entry for entry in a.datas if not _should_drop(entry)])
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="deviceMngUI",
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
    name="deviceMngUI",
)
