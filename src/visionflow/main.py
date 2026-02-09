"""VisionFlow Sample Launcher"""

from __future__ import annotations

import importlib
import sys


SAMPLES = [
    {
        "category": "Camera",
        "items": [
            ("camera.simple", "visionflow.sample.camera.simple", "기본 카메라 뷰어"),
            ("camera.dual_cam", "visionflow.sample.camera.dual_cam", "듀얼 카메라"),
            ("camera.list_cameras", "visionflow.sample.camera.list_cameras", "카메라 디바이스 목록"),
            ("camera.list_resolutions", "visionflow.sample.camera.list_resolutions", "카메라 해상도 목록"),
        ],
    },
    {
        "category": "Face Detection",
        "items": [
            ("face_detection.simple", "visionflow.sample.face_detection.simple", "얼굴 검출"),
            ("face_detection.simple_landmark", "visionflow.sample.face_detection.simple_landmark", "얼굴 랜드마크"),
            ("face_detection.transform_3d", "visionflow.sample.face_detection.transform_3d", "3D 얼굴 변환"),
        ],
    },
    {
        "category": "Pose",
        "items": [
            ("pose.simple", "visionflow.sample.pose.simple", "포즈 감지"),
        ],
    },
    {
        "category": "Full Pipeline",
        "items": [
            ("detect_test", "visionflow.sample.detect_test", "전체 파이프라인 테스트"),
        ],
    },
]


def _print_menu():
    print()
    print("=" * 50)
    print("  VisionFlow Sample Launcher")
    print("=" * 50)

    idx = 1
    index_map = {}
    for group in SAMPLES:
        print(f"\n  [{group['category']}]")
        for name, module, desc in group["items"]:
            print(f"    {idx}) {name:<30s} - {desc}")
            index_map[idx] = (name, module)
            idx += 1

    print(f"\n    0) 종료")
    print()
    return index_map


def main():
    # 런처에서 실행할 때 sys.argv를 리셋하여
    # 각 샘플의 argparse가 올바르게 동작하도록 함
    original_argv = sys.argv[:]

    while True:
        index_map = _print_menu()

        try:
            choice = input("  선택: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if choice == "0" or choice.lower() == "q":
            break

        try:
            num = int(choice)
        except ValueError:
            print(f"  [!] 올바른 번호를 입력하세요.")
            continue

        if num not in index_map:
            print(f"  [!] 1~{len(index_map)} 또는 0을 입력하세요.")
            continue

        name, module_path = index_map[num]
        print(f"\n  >>> {name} 실행 중...\n")

        try:
            # sys.argv를 리셋하여 각 샘플의 argparse가 정상 동작하도록 함
            sys.argv = [name]
            mod = importlib.import_module(module_path)
            mod.main()
        except KeyboardInterrupt:
            print(f"\n  [{name}] 중단됨")
        except Exception as e:
            print(f"\n  [{name}] 오류: {e}")
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    main()
