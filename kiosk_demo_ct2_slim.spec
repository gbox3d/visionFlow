# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.building.datastruct import TOC
from PyInstaller.utils.hooks import collect_dynamic_libs


project_root = Path(globals().get("SPECPATH", ".")).resolve()
src_root = project_root / "src"


def _add_data_file(out: list[tuple[str, str]], relative_path: str, target_dir: str | None = None) -> None:
    file_path = project_root / relative_path
    if not file_path.exists() or not file_path.is_file():
        return
    out_dir = target_dir if target_dir is not None else str(Path(relative_path).parent).replace("\\", "/")
    out.append((str(file_path), out_dir))


datas = []
for rel_model_path in (
    "models/blaze_face_short_range.tflite",
    "models/face_landmarker.task",
    "models/pose_landmarker.task",
    "models/pose_landmarker_lite.task",
):
    _add_data_file(datas, rel_model_path)

_add_data_file(datas, "font/DungGeunMo.ttf")
if (project_root / ".env").exists():
    datas.append((str(project_root / ".env"), "."))

binaries = []
binaries += collect_dynamic_libs("cv2")
binaries += collect_dynamic_libs("mediapipe")
binaries += collect_dynamic_libs("ctranslate2")

_DROP_BINARY_HINTS = {
    "opencv_videoio_ffmpeg",
    "pyside6/",
}


def _keep_binary(entry) -> bool:
    dst = str(entry[0]).replace("\\", "/").lower()
    src = str(entry[1]).replace("\\", "/").lower() if len(entry) > 1 else ""
    probe = f"{dst}|{src}"
    return not any(hint in probe for hint in _DROP_BINARY_HINTS)


hiddenimports = [
    "cv2_enumerate_cameras",
    "faster_whisper",
    "faster_whisper.transcribe",
    "voiceFlow.vendors.miso_stt.backends",
    "voiceFlow.vendors.miso_stt.backends.ct2",
]

excludes = [
    "PySide6",
    "shiboken6",
    "visionflow.sample",
    "voiceFlow.sample",
    "sklearn",
    "numba",
    "llvmlite",
    "onnxruntime",
    "librosa",
    "scipy",
    "hf_xet",
    "av",
    "voiceFlow.vendors.miso_stt.backends.hf_generate",
    "voiceFlow.vendors.miso_stt.backends.hf_pipeline",
    "transformers",
    "accelerate",
    "torch",
    "torchaudio",
    "torchvision",
]


a = Analysis(
    ["main.py"],
    pathex=[str(project_root), str(src_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={"matplotlib": {"backends": "Agg"}},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
a.binaries = TOC([entry for entry in a.binaries if _keep_binary(entry)])
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="kiosk_demo_ct2_slim",
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
    name="kiosk_demo_ct2_slim",
)

