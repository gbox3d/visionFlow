from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
from cv2_enumerate_cameras import enumerate_cameras


# -------------------------------
# 설정값
# -------------------------------
REQUEST_W = 1920
REQUEST_H = 1080
READ_TRIALS = 3
OUT_JSON = "camera_map.json"


# -------------------------------
# enumerate index는 반드시 auto backend
# -------------------------------
def probe_camera(idx: int) -> int:
    cap = cv2.VideoCapture(idx)  # ❗ DSHOW 절대 사용 금지
    score = 0

    try:
        if not cap.isOpened():
            return 0
        score += 1

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_H)
        time.sleep(0.1)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w >= REQUEST_W and h >= REQUEST_H:
            score += 1

        for _ in range(READ_TRIALS):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            score += 1

    finally:
        cap.release()

    return score


# -------------------------------
# 연속 index 기준으로 분리
# -------------------------------
def split_by_continuous_indices(indices: List[int]) -> List[List[int]]:
    if not indices:
        return []

    indices = sorted(indices)
    groups: List[List[int]] = []
    current = [indices[0]]

    for prev, cur in zip(indices, indices[1:]):
        if cur == prev + 1:
            current.append(cur)
        else:
            groups.append(current)
            current = [cur]

    groups.append(current)
    return groups


def main():
    cameras = enumerate_cameras()

    # 1차 그룹: (vid, pid, name)
    groups: Dict[Tuple[int | None, int | None, str], List[int]] = defaultdict(list)
    for cam in cameras:
        groups[(cam.vid, cam.pid, cam.name)].append(cam.index)

    result = []

    # 2차: 연속 index 묶음 = 물리 카메라
    for (vid, pid, name), indices in groups.items():
        print(f"\n[MODEL] {name} vid={vid} pid={pid}")
        sub_groups = split_by_continuous_indices(indices)

        for sub in sub_groups:
            print(f"  [DEVICE] entries={sub}")

            best_idx = None
            best_score = -1

            for idx in sub:
                score = probe_camera(idx)
                print(f"    index {idx} -> score {score}")
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None and best_score > 0:
                result.append(
                    {
                        "name": name,
                        "vid": vid,
                        "pid": pid,
                        "camera_id": best_idx,
                        "backend": "auto",
                        "entries": sub,
                        "score": best_score,
                    }
                )
                print(f"    => SELECT {best_idx}")

    # 결과 저장
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n=== camera_map.json ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
