# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules


project_root = Path(globals().get("SPECPATH", ".")).resolve()
src_root = project_root / "src"


def _collect_files_under(relative_dir: str):
    base = project_root / relative_dir
    if not base.exists() or not base.is_dir():
        return []
    out = []
    for file_path in base.rglob("*"):
        if not file_path.is_file():
            continue
        target_dir = str(file_path.parent.relative_to(project_root))
        out.append((str(file_path), target_dir))
    return out


datas = []
datas.extend(_collect_files_under("models"))
datas.extend(_collect_files_under("font"))
if (project_root / ".env").exists():
    datas.append((str(project_root / ".env"), "."))

binaries = []
binaries += collect_dynamic_libs("cv2")
binaries += collect_dynamic_libs("mediapipe")
binaries += collect_dynamic_libs("torch")

hiddenimports = []
hiddenimports += collect_submodules("visionflow")
hiddenimports += collect_submodules("voiceFlow")


a = Analysis(
    ["main.py"],
    pathex=[str(project_root), str(src_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
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
    name="kiosk_demo",
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
    name="kiosk_demo",
)
