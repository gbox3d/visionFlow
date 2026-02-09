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


def main():
    
    
    suppress_opencv_warnings()
    
    parser = argparse.ArgumentParser(
        description=(
            "visionflow camera device lister\n\n"
            "OpenCV를 사용하여 현재 시스템에서 접근 가능한\n"
            "카메라 디바이스 목록을 출력합니다.\n\n"
            "- Windows / Linux 공통 사용 가능\n"
            "- 이후 CameraSource 설정의 기준으로 사용"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--max-devices",
        type=int,
        default=10,
        help="탐색할 최대 카메라 디바이스 번호 (기본값: 10)",
    )
    parser.add_argument(
        "--dshow",
        type=int,
        default=1,
        help="Windows에서 DirectShow(CAP_DSHOW) 사용 여부 (1=사용, 0=자동)",
    )

    args = parser.parse_args()

    print("=== visionflow camera device list ===")
    print(f"scan range : 0 ~ {args.max_devices - 1}")
    print(f"CAP_DSHOW  : {'ON' if args.dshow else 'AUTO'}")
    print("------------------------------------")

    found = False
    for cam_id in range(args.max_devices):
        ok = probe_camera(cam_id, use_dshow=(args.dshow != 0))
        status = "OK" if ok else "--"
        print(f"camera_id {cam_id:2d} : {status}")
        if ok:
            found = True

    if not found:
        print("\n(no available camera devices found)")


if __name__ == "__main__":
    main()
