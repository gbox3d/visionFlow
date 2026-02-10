"""
file : sample/camera/list_cameras.py

description:
    OpenCV 기반으로 사용 가능한 카메라 디바이스 목록을 출력하는 예제

important:
    dont not edit this commented block
"""

from __future__ import annotations

import argparse
import cv2

from visionflow.utils.etc import suppress_opencv_warnings


def probe_camera(camera_id: int, use_dshow: bool) -> bool:
    """
    지정한 camera_id가 열리는지 테스트
    """
    if use_dshow:
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        cap.release()
        return False

    ret, _ = cap.read()
    cap.release()
    return bool(ret)


def _enumerate_with_hwinfo():
    """
    cv2-enumerate-cameras를 사용하여 하드웨어 정보 포함 카메라 목록 출력
    """
    from cv2_enumerate_cameras import enumerate_cameras

    cameras = list(enumerate_cameras())

    if not cameras:
        print("(no camera devices found)")
        return

    # index에 backend가 인코딩됨 (DSHOW=700+n, MSMF=1400+n)
    backend_ranges = {
        cv2.CAP_DSHOW: "DSHOW",
        cv2.CAP_MSMF: "MSMF",
    }

    for cam in cameras:
        # backend 판별: cam.backend가 0인 경우 index에서 추출
        if cam.backend and cam.backend in backend_ranges:
            backend_name = backend_ranges[cam.backend]
            cam_id = cam.index % 100
        else:
            # index에서 backend 추론
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


def _probe_scan(max_devices: int, use_dshow: bool):
    """
    기존 방식: 인덱스 0~N을 순회하며 열리는지 확인
    """
    print(f"scan range : 0 ~ {max_devices - 1}")
    print(f"CAP_DSHOW  : {'ON' if use_dshow else 'AUTO'}")
    print()

    found = False
    for cam_id in range(max_devices):
        ok = probe_camera(cam_id, use_dshow=use_dshow)
        status = "OK" if ok else "--"
        print(f"  camera_id {cam_id:2d} : {status}")
        if ok:
            found = True

    if not found:
        print("\n(no available camera devices found)")


def main():
    suppress_opencv_warnings()

    parser = argparse.ArgumentParser(
        description=(
            "visionflow camera device lister\n\n"
            "카메라 디바이스 목록을 출력합니다.\n"
            "기본: 하드웨어 정보 포함 (cv2-enumerate-cameras)\n"
            "--probe 옵션: 인덱스 순회 방식 (OpenCV 직접 탐색)"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--probe",
        action="store_true",
        help="인덱스 순회 방식으로 탐색 (하드웨어 정보 없음)",
    )
    parser.add_argument(
        "--max-devices",
        type=int,
        default=10,
        help="--probe 모드에서 탐색할 최대 디바이스 번호 (기본값: 10)",
    )
    parser.add_argument(
        "--dshow",
        type=int,
        default=1,
        help="Windows에서 DirectShow(CAP_DSHOW) 사용 여부 (1=사용, 0=자동)",
    )

    args = parser.parse_args()

    print("=== visionflow camera device list ===")
    print()

    if args.probe:
        _probe_scan(args.max_devices, use_dshow=(args.dshow != 0))
    else:
        _enumerate_with_hwinfo()


if __name__ == "__main__":
    main()
