# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files

project_root = os.path.abspath(".")
src_root = os.path.join(project_root, "src")

def collect_dir_as_datas(src_dir: str, dest_root: str):
    """
    src_dir 아래 모든 파일을 찾아 datas[(src_file, dest_dir)] 형태로 만든다.
    PyInstaller Analysis(datas=...)는 (src, dest) 2튜플만 안전하게 받는다.
    """
    datas = []
    src_dir = os.path.abspath(src_dir)

    if not os.path.isdir(src_dir):
        return datas

    for root, _, files in os.walk(src_dir):
        rel_dir = os.path.relpath(root, src_dir)
        # dest는 "디렉토리"여야 함
        dest_dir = dest_root if rel_dir == "." else os.path.join(dest_root, rel_dir)

        for fn in files:
            src_file = os.path.join(root, fn)
            datas.append((src_file, dest_dir))
    return datas

def normalize_hook_datas(items):
    """
    PyInstaller/훅 버전에 따라 (src,dest) 또는 (src,dest,typecode)로 올 수 있어
    Analysis 단계에서는 (src,dest)로 정규화.
    """
    out = []
    for t in items:
        if len(t) == 2:
            src, dest = t
        elif len(t) == 3:
            src, dest, _ = t
        else:
            raise ValueError(f"Unexpected datas tuple: {t}")
        out.append((src, dest))
    return out

# 1) 우리 프로젝트 리소스(모델/폰트) → 반드시 2튜플로
project_datas = []
project_datas += collect_dir_as_datas("models", "models")
project_datas += collect_dir_as_datas("font", "font")

# 2) mediapipe 데이터도 Analysis에서는 2튜플로 정규화해서 넣기
mp_datas = normalize_hook_datas(collect_data_files("mediapipe"))

a = Analysis(
    [os.path.join(src_root, 'visionflow', 'sample', 'detect_test.py')],
    pathex=[src_root],
    binaries=[],
    datas=project_datas + mp_datas,
    hiddenimports=[
        'visionflow',
        'visionflow.pipeline',
        'visionflow.sources',
        'visionflow.workers',
        'visionflow.utils',
        'visionflow.processors',
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
    a.binaries,
    a.zipfiles,
    a.datas,
    name='detect_test_app',
    debug=False,
    strip=False,
    upx=True,
    console=False,
)
