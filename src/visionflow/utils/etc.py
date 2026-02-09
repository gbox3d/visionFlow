"""
file : visionFlow/utils/etc.py
author: gbox3d

important:
do not edit this commented block

카메라 관련 정보를 얻어내는 함수들 모아놓은 모듈

디바이스 목록
디바이스 해상도 정보
디바이스 포맷 정보
등등

"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2

import numpy as np
import pygame


import sys
import os

def resource_path(relative_path):
    """ 실행 파일 환경과 개발 환경 모두에서 올바른 경로를 반환합니다. """
    try:
        # PyInstaller에 의해 임시폴더(_MEIPASS)가 생성된 경우
        base_path = sys._MEIPASS
    except Exception:
        # 일반 파이썬 실행 환경인 경우 (현재 폴더)
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# -------------------------
# Dataclasses
# -------------------------
@dataclass(frozen=True)
class CameraInfo:
    index: int
    opened: bool
    backend: str
    request_size: Tuple[int, int]
    actual_size: Tuple[int, int]
    fps: float
    fourcc: str


@dataclass(frozen=True)
class ResolutionProbe:
    request_size: Tuple[int, int]
    actual_size: Tuple[int, int]
    ok: bool


@dataclass(frozen=True)
class FormatProbe:
    request_fourcc: str
    actual_fourcc: str
    ok: bool


# -------------------------
# Backend helpers
# -------------------------
def backend_code(name: str) -> int:
    name = (name or "").lower()
    if name in ("dshow", "directshow"):
        return cv2.CAP_DSHOW
    if name in ("msmf", "mediafoundation"):
        # cv2가 빌드에 따라 없을 수 있으니 hasattr 체크
        if hasattr(cv2, "CAP_MSMF"):
            return cv2.CAP_MSMF
        return 0
    if name in ("any", "auto", "default"):
        return 0
    raise ValueError(f"unknown backend: {name} (use: dshow|msmf|any)")


def backend_label(code: int) -> str:
    if code == cv2.CAP_DSHOW:
        return "DSHOW"
    if hasattr(cv2, "CAP_MSMF") and code == cv2.CAP_MSMF:
        return "MSMF"
    return "ANY"


# -------------------------
# FourCC helpers
# -------------------------
def fourcc_to_str(v: float) -> str:
    """
    OpenCV get(CAP_PROP_FOURCC)는 float로 나오는 경우가 많음.
    int로 변환 후 4글자 문자열로 복원.
    """
    try:
        i = int(v)
    except Exception:
        return "----"
    chars = [chr((i >> (8 * k)) & 0xFF) for k in range(4)]
    s = "".join(chars)
    # 비가시 문자 방지
    return "".join(c if 32 <= ord(c) <= 126 else "-" for c in s)


def str_to_fourcc(s: str) -> int:
    s = (s or "----")[:4].ljust(4, "-")
    return cv2.VideoWriter_fourcc(*s)


# -------------------------
# Low-level capture / probe
# -------------------------
def _open_cap(index: int, backend: int) -> cv2.VideoCapture:
    return cv2.VideoCapture(index, backend)


def _safe_grab_retrieve(cap: cv2.VideoCapture, warmup_grab: int = 5, timeout_s: float = 1.2) -> Optional[Tuple[int, int]]:
    """
    Windows에서 read()가 멎는 케이스가 있어 grab/retrieve로 가볍게 확인.
    성공하면 (w,h) 반환, 실패면 None.
    """
    start = time.time()
    for _ in range(max(0, warmup_grab)):
        if time.time() - start > timeout_s:
            return None
        if not cap.grab():
            return None
        time.sleep(0.02)

    ok, frame = cap.retrieve()
    if not ok or frame is None or getattr(frame, "size", 0) == 0:
        return None

    h, w = frame.shape[:2]
    return (w, h)


def probe_camera(
    index: int,
    backend: int = cv2.CAP_DSHOW,
    request_size: Tuple[int, int] = (1280, 720),
    request_fourcc: Optional[str] = None,
    warmup_grab: int = 5,
    timeout_s: float = 1.2,
) -> CameraInfo:
    """
    카메라 1개를 '열 수 있는지' + 요청 해상도 적용 후 실제 해상도/포맷 정보 확인.
    """
    w_req, h_req = request_size
    cap = _open_cap(index, backend)

    if not cap.isOpened():
        return CameraInfo(
            index=index,
            opened=False,
            backend=backend_label(backend),
            request_size=(w_req, h_req),
            actual_size=(0, 0),
            fps=0.0,
            fourcc="----",
        )

    # 해상도 요청
    if w_req > 0 and h_req > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w_req))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h_req))

    # 포맷(FourCC) 요청(가능하면)
    if request_fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, float(str_to_fourcc(request_fourcc)))

    size = _safe_grab_retrieve(cap, warmup_grab=warmup_grab, timeout_s=timeout_s)
    if size is None:
        cap.release()
        return CameraInfo(
            index=index,
            opened=False,
            backend=backend_label(backend),
            request_size=(w_req, h_req),
            actual_size=(0, 0),
            fps=0.0,
            fourcc="----",
        )

    # FPS는 장치/드라이버에 따라 0이거나 부정확할 수 있음(참고용)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 0 else 0.0

    actual_fourcc = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))

    cap.release()
    return CameraInfo(
        index=index,
        opened=True,
        backend=backend_label(backend),
        request_size=(w_req, h_req),
        actual_size=size,
        fps=fps,
        fourcc=actual_fourcc,
    )


def scan_cameras(
    max_index: int = 10,
    backend: int = cv2.CAP_DSHOW,
    request_size: Tuple[int, int] = (1280, 720),
    warmup_grab: int = 5,
) -> List[CameraInfo]:
    """
    0..max_index 까지 카메라를 스캔
    """
    out: List[CameraInfo] = []
    for i in range(0, max(0, max_index) + 1):
        out.append(
            probe_camera(
                index=i,
                backend=backend,
                request_size=request_size,
                request_fourcc=None,
                warmup_grab=warmup_grab,
            )
        )
    return out

def generate_resolution_candidates(
    max_w: int = 3840,
    max_h: int = 2160,
    include_43: bool = True,
    include_169: bool = True,
    include_1610: bool = True,
) -> List[Tuple[int, int]]:
    """
    크로스플랫폼(OpenCV)에서 쓸 해상도 후보 리스트 생성기.
    - OpenCV로 '지원 목록'을 직접 얻을 수 없으니, 흔한 해상도를 후보로 만들어 probe 한다.
    """
    s = set()

    if include_43:
        for w, h in [(320, 240), (640, 480), (800, 600), (1024, 768), (1280, 960), (1600, 1200), (2048, 1536)]:
            if w <= max_w and h <= max_h:
                s.add((w, h))

    if include_169:
        for w, h in [(640, 360), (854, 480), (960, 540), (1280, 720), (1600, 900),
                     (1920, 1080), (2560, 1440), (3840, 2160)]:
            if w <= max_w and h <= max_h:
                s.add((w, h))

    if include_1610:
        for w, h in [(1280, 800), (1440, 900), (1680, 1050), (1920, 1200)]:
            if w <= max_w and h <= max_h:
                s.add((w, h))

    return sorted(s, key=lambda x: (x[0], x[1]))


def probe_resolutions(
    index: int,
    backend: int = cv2.CAP_DSHOW,
    candidates: Optional[Iterable[Tuple[int, int]]] = None,
    warmup_grab: int = 5,
    timeout_s: float = 1.2,
    accept_tolerance_px: int = 2,
) -> List[ResolutionProbe]:
    """
    후보 해상도들을 실제로 set + 프레임 수신으로 판정(probe).
    - ok: 실제 해상도가 요청과 거의 같을 때(±accept_tolerance_px)
    """
    if candidates is None:
        candidates = generate_resolution_candidates()

    results: List[ResolutionProbe] = []
    for (w, h) in candidates:
        info = probe_camera(
            index=index,
            backend=backend,
            request_size=(w, h),
            request_fourcc=None,
            warmup_grab=warmup_grab,
            timeout_s=timeout_s,
        )
        if not info.opened:
            results.append(ResolutionProbe((w, h), (0, 0), False))
            continue

        aw, ah = info.actual_size
        ok = (abs(aw - w) <= accept_tolerance_px) and (abs(ah - h) <= accept_tolerance_px)
        results.append(ResolutionProbe((w, h), (aw, ah), ok))

    return results

def list_supported_resolutions(
    index: int,
    backend: int = cv2.CAP_DSHOW,
    candidates: Optional[Iterable[Tuple[int, int]]] = None,
    warmup_grab: int = 5,
    timeout_s: float = 1.2,
    accept_tolerance_px: int = 2,
) -> List[Tuple[int, int]]:
    """
    디바이스 1개에 대해 '지원 해상도 목록(프로빙 기반)'을 반환.
    - OK로 판정된 요청 해상도만 모아 정렬하여 반환
    """
    probes = probe_resolutions(
        index=index,
        backend=backend,
        candidates=candidates,
        warmup_grab=warmup_grab,
        timeout_s=timeout_s,
        accept_tolerance_px=accept_tolerance_px,
    )
    ok = [p.request_size for p in probes if p.ok]
    # 중복 제거 + 정렬
    ok = sorted(set(ok), key=lambda x: (x[0], x[1]))
    return ok


def list_supported_resolutions_relaxed(
    index: int,
    backend: int = cv2.CAP_DSHOW,
    candidates: Optional[Iterable[Tuple[int, int]]] = None,
    warmup_grab: int = 5,
    timeout_s: float = 1.2,
) -> List[Tuple[int, int]]:
    """
    드라이버가 set()을 무시하거나 스케일링해서 exact match가 잘 안 나는 카메라가 있습니다.
    이 함수는:
      - 프레임만 정상으로 나오면 "actual 해상도"를 수집하여 목록으로 반환합니다.
    즉, exact match가 아니라도 "해당 디바이스가 실제로 낼 수 있는 해상도들"을 모읍니다.
    """
    if candidates is None:
        candidates = generate_resolution_candidates()

    actuals: List[Tuple[int, int]] = []
    for (w, h) in candidates:
        info = probe_camera(
            index=index,
            backend=backend,
            request_size=(w, h),
            request_fourcc=None,
            warmup_grab=warmup_grab,
            timeout_s=timeout_s,
        )
        if info.opened and info.actual_size != (0, 0):
            actuals.append(info.actual_size)

    return sorted(set(actuals), key=lambda x: (x[0], x[1]))


def probe_formats(
    index: int,
    backend: int = cv2.CAP_DSHOW,
    request_size: Tuple[int, int] = (1280, 720),
    candidates: Optional[Iterable[str]] = None,
    warmup_grab: int = 5,
    timeout_s: float = 1.2,
) -> List[FormatProbe]:
    """
    포맷(FourCC)도 OpenCV로 "지원목록"을 직접 열람하기가 어렵습니다.
    -> 후보 FourCC를 set하고, 실제 get 값 + 프레임 수신 성공 여부로 probe.
    """
    if candidates is None:
        candidates = [
            "MJPG",
            "YUY2",
            "H264",
            "NV12",
            "YV12",
        ]

    out: List[FormatProbe] = []
    for fcc in candidates:
        info = probe_camera(
            index=index,
            backend=backend,
            request_size=request_size,
            request_fourcc=fcc,
            warmup_grab=warmup_grab,
            timeout_s=timeout_s,
        )
        # 열린 다음에도 실제 fourcc가 바뀌지 않거나 드라이버가 무시할 수 있음
        ok = info.opened and (info.fourcc == fcc)
        out.append(FormatProbe(request_fourcc=fcc, actual_fourcc=info.fourcc, ok=ok))
    return out


# -------------------------
# Pretty print helpers
# -------------------------
def print_camera_table(infos: List[CameraInfo]) -> None:
    print("")
    print("=== Camera Devices ===")
    print(f"{'IDX':>3} | {'OPEN':>4} | {'BACK':>5} | {'REQ':>11} | {'ACTUAL':>11} | {'FPS':>6} | {'FCC':>4}")
    print("-" * 72)
    for it in infos:
        rw, rh = it.request_size
        aw, ah = it.actual_size
        print(
            f"{it.index:>3} | {str(it.opened):>4} | {it.backend:>5} | "
            f"{rw:>4}x{rh:<6} | {aw:>4}x{ah:<6} | {it.fps:>6.1f} | {it.fourcc:>4}"
        )
    print("")


def print_resolution_probes(probes: List[ResolutionProbe]) -> None:
    print("  - Resolutions (probe)")
    for p in probes:
        rw, rh = p.request_size
        aw, ah = p.actual_size
        mark = "OK " if p.ok else "   "
        print(f"    {mark} req={rw}x{rh} -> actual={aw}x{ah}")


def print_format_probes(probes: List[FormatProbe]) -> None:
    print("  - Formats (probe)")
    for p in probes:
        mark = "OK " if p.ok else "   "
        print(f"    {mark} req={p.request_fourcc} -> actual={p.actual_fourcc}")

# -------------------------
# opencv helpers
# -------------------------
"""
OpenCV 경고 메시지 억제
"""
def suppress_opencv_warnings():
    # OpenCV 4.5+에서 동작하는 경우가 많음
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        return
    except Exception:
        pass

    # OpenCV 4.8+ (일부 빌드)에서 제공
    try:
        cv2.setLogLevel(3)  # 0=verbose, 1=info, 2=warning, 3=error, 4=fatal, 5=silent(빌드에 따라 다름)
        return
    except Exception:
        pass

    # 더 최신 API (있는 경우)
    try:
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
        return
    except Exception:
        pass