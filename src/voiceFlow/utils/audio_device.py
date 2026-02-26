from __future__ import annotations

import re
from typing import Optional


def _normalize_device_path_key(path: str) -> str:
    normalized = (path or "").strip().lower()
    if "#{" in normalized:
        return normalized.split("#{")[0]
    return normalized


def parse_mic_device(value: str) -> Optional[int | str]:
    parsed = (value or "").strip()
    if not parsed:
        return None
    try:
        return int(parsed)
    except ValueError:
        return parsed


def _normalize_audio_name_key(name: str) -> str:
    normalized = (name or "").strip().lower()
    normalized = normalized.replace("\r", " ").replace("\n", " ")
    return re.sub(r"[^0-9a-zA-Z가-힣]+", "", normalized)


def _tokenize_audio_name(name: str) -> set[str]:
    cleaned = (name or "").strip().lower()
    cleaned = cleaned.replace("\r", " ").replace("\n", " ")
    tokens = re.split(r"[^0-9a-zA-Z가-힣]+", cleaned)
    return {t for t in tokens if t}


def resolve_camera_name_from_path(device_path: str) -> Optional[str]:
    target_key = _normalize_device_path_key(device_path)
    if not target_key:
        return None

    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except Exception:
        return None

    for cam in enumerate_cameras():
        cam_path = getattr(cam, "path", None)
        cam_name = getattr(cam, "name", None)
        if not cam_path or not cam_name:
            continue
        if _normalize_device_path_key(str(cam_path)) == target_key:
            name = str(cam_name).strip()
            if name:
                return name
    return None


def resolve_mic_device_from_camera_path(device_path: str) -> Optional[int | str]:
    camera_name = resolve_camera_name_from_path(device_path)
    if not camera_name:
        return None

    try:
        import sounddevice as sd
    except Exception:
        return None

    camera_name_lower = camera_name.lower()
    camera_name_key = _normalize_audio_name_key(camera_name)
    camera_tokens = _tokenize_audio_name(camera_name)
    camera_tokens = {t for t in camera_tokens if len(t) >= 3}

    try:
        devices = sd.query_devices()
    except Exception:
        return None

    scored: list[tuple[int, int]] = []
    for idx, device in enumerate(devices):
        max_input_channels = int(device.get("max_input_channels", 0))
        if max_input_channels <= 0:
            continue

        device_name = str(device.get("name", "")).strip()
        if not device_name:
            continue

        device_name_lower = device_name.lower()
        device_name_key = _normalize_audio_name_key(device_name)
        device_tokens = _tokenize_audio_name(device_name)

        score = -1

        # Strong matches first.
        if device_name_lower == camera_name_lower or device_name_key == camera_name_key:
            score = 400
        elif camera_name_lower in device_name_lower or device_name_lower in camera_name_lower:
            score = 300
        elif camera_name_key and (camera_name_key in device_name_key or device_name_key in camera_name_key):
            score = 260
        else:
            overlap = len(camera_tokens & device_tokens)
            if overlap >= 2:
                score = 200 + overlap
            elif overlap == 1:
                score = 120

        if score >= 0:
            scored.append((score, idx))

    if not scored:
        # Important: None triggers caller fallback to MIC_DEVICE/default.
        return None

    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    return scored[0][1]
