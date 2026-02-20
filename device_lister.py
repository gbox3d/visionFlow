from __future__ import annotations

import argparse

import cv2


def _suppress_opencv_warnings() -> None:
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        return
    except Exception:
        pass
    try:
        cv2.setLogLevel(3)
        return
    except Exception:
        pass
    try:
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
    except Exception:
        pass


def _probe_camera(camera_id: int, use_dshow: bool) -> bool:
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW) if use_dshow else cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        cap.release()
        return False
    ok, _ = cap.read()
    cap.release()
    return bool(ok)


def _print_camera_list_hwinfo() -> None:
    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except Exception as exc:
        print(f"[camera] cv2-enumerate-cameras import failed: {exc}")
        return

    cameras = list(enumerate_cameras())
    if not cameras:
        print("[camera] no camera devices found")
        return

    backend_ranges = {
        cv2.CAP_DSHOW: "DSHOW",
        cv2.CAP_MSMF: "MSMF",
    }

    print("=== camera devices (hw info) ===")
    print()
    for cam in cameras:
        if cam.backend and cam.backend in backend_ranges:
            backend_name = backend_ranges[cam.backend]
            cam_id = cam.index % 100
        else:
            matched = False
            for base, name in sorted(backend_ranges.items(), reverse=True):
                if cam.index >= base:
                    backend_name = name
                    cam_id = cam.index - base
                    matched = True
                    break
            if not matched:
                backend_name = "AUTO"
                cam_id = cam.index

        vid = f"{cam.vid:04X}" if cam.vid else "----"
        pid = f"{cam.pid:04X}" if cam.pid else "----"

        print(f"  camera_id : {cam_id}")
        print(f"  name      : {cam.name}")
        print(f"  VID:PID   : {vid}:{pid}")
        print(f"  backend   : {backend_name}")
        if cam.path:
            print(f"  path      : {cam.path}")
        print()


def _print_camera_list_probe(max_devices: int, use_dshow: bool) -> None:
    print("=== camera devices (probe mode) ===")
    print(f"scan range : 0 ~ {max(0, max_devices - 1)}")
    print(f"CAP_DSHOW  : {'ON' if use_dshow else 'AUTO'}")
    print()

    found = False
    for cam_id in range(max(0, max_devices)):
        ok = _probe_camera(cam_id, use_dshow=use_dshow)
        status = "OK" if ok else "--"
        print(f"  camera_id {cam_id:2d} : {status}")
        found = found or ok

    if not found:
        print("\n[camera] no available camera devices found")
    print()


def _print_mic_list() -> None:
    try:
        import sounddevice as sd
    except Exception as exc:
        print(f"[mic] sounddevice import failed: {exc}")
        return

    try:
        devices = sd.query_devices()
    except Exception as exc:
        print(f"[mic] failed to query devices: {exc}")
        return

    print("=== microphone devices ===")
    print()
    found = False
    for idx, device in enumerate(devices):
        max_input = int(device.get("max_input_channels", 0))
        if max_input <= 0:
            continue
        hostapi_index = int(device.get("hostapi", -1))
        hostapi_name = "-"
        try:
            hostapi_name = str(sd.query_hostapis(hostapi_index).get("name", "-"))
        except Exception:
            pass
        print(f"  mic_id    : {idx}")
        print(f"  name      : {device.get('name', '-')}")
        print(f"  hostapi   : {hostapi_name}")
        print(f"  channels  : {max_input}")
        print()
        found = True

    if not found:
        print("[mic] no input devices found\n")

    try:
        default_input = sd.query_devices(kind="input")
        hostapi_name = "-"
        try:
            hostapi_name = str(sd.query_hostapis(int(default_input.get("hostapi", -1))).get("name", "-"))
        except Exception:
            pass
        print("default input:")
        print(f"  mic_id    : {default_input.get('index', '-')}")
        print(f"  name      : {default_input.get('name', '-')}")
        print(f"  hostapi   : {hostapi_name}")
    except Exception as exc:
        print(f"default input: not available ({exc})")
    print()


def main() -> None:
    _suppress_opencv_warnings()

    parser = argparse.ArgumentParser(
        description=(
            "Device lister (camera + microphone)\n"
            "- default: camera hwinfo + microphone list\n"
            "- --probe: camera index probing mode"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--probe", action="store_true", help="camera probe mode")
    parser.add_argument("--max-devices", type=int, default=10, help="max camera index in probe mode")
    parser.add_argument("--dshow", type=int, default=1, help="camera probe backend (1=CAP_DSHOW, 0=auto)")
    parser.add_argument("--camera-only", action="store_true", help="print only camera devices")
    parser.add_argument("--mic-only", action="store_true", help="print only microphone devices")
    args = parser.parse_args()

    if args.camera_only and args.mic_only:
        raise SystemExit("Cannot use --camera-only and --mic-only together.")

    print("=== NeuroFlow Device Lister ===")
    print()

    if not args.mic_only:
        if args.probe:
            _print_camera_list_probe(max_devices=args.max_devices, use_dshow=(args.dshow != 0))
        else:
            _print_camera_list_hwinfo()

    if not args.camera_only:
        _print_mic_list()


if __name__ == "__main__":
    main()

