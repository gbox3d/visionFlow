# sample/list_resolutions.py
from __future__ import annotations

import argparse

from visionflow.utils.etc import backend_code, generate_resolution_candidates, list_supported_resolutions, list_supported_resolutions_relaxed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-id", type=int, default=0)
    ap.add_argument("--backend", type=str, default="any", help="any|dshow|msmf|v4l2 (가능하면 any 권장)")
    ap.add_argument("--max-w", type=int, default=3840)
    ap.add_argument("--max-h", type=int, default=2160)
    ap.add_argument("--relaxed", type=int, default=0, help="1이면 actual 해상도 수집 모드(스케일링 장치용)")
    args = ap.parse_args()

    be = backend_code(args.backend)

    candidates = generate_resolution_candidates(max_w=args.max_w, max_h=args.max_h)

    if args.relaxed:
        modes = list_supported_resolutions_relaxed(args.camera_id, backend=be, candidates=candidates)
        print(f"[relaxed] camera={args.camera_id} backend={args.backend} -> actual modes ({len(modes)})")
    else:
        modes = list_supported_resolutions(args.camera_id, backend=be, candidates=candidates)
        print(f"[exact] camera={args.camera_id} backend={args.backend} -> supported req modes ({len(modes)})")

    for (w, h) in modes:
        print(f" - {w}x{h}")


if __name__ == "__main__":
    main()
