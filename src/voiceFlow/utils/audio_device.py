from __future__ import annotations

from typing import Optional

from visionflow.utils.etc import normalize_device_path_key


def parse_mic_device(value: str) -> Optional[int | str]:
    parsed = (value or "").strip()
    if not parsed:
        return None
    try:
        return int(parsed)
    except ValueError:
        return parsed


def resolve_camera_name_from_path(device_path: str) -> Optional[str]:
    target_key = normalize_device_path_key(device_path)
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
        if normalize_device_path_key(str(cam_path)) == target_key:
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
        return camera_name

    camera_name_lower = camera_name.lower()
    try:
        devices = sd.query_devices()
    except Exception:
        return camera_name

    for idx, device in enumerate(devices):
        max_input_channels = int(device.get("max_input_channels", 0))
        if max_input_channels <= 0:
            continue
        device_name = str(device.get("name", ""))
        if camera_name_lower in device_name.lower():
            return idx

    return camera_name
