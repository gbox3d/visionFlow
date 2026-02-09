from __future__ import annotations

import time
from typing import Any, Dict, Optional

import cv2
import mediapipe as mp

from visionflow.pipeline.packet import FramePacket


class FaceDetectorProcessor:
    """
    MediaPipe Tasks API 기반 FaceDetector (동기)
    - running_mode: IMAGE 또는 VIDEO 권장
    - process(pkt) -> FramePacket(roi 포함)
    """

    def __init__(
        self,
        model_path: str = "models/blaze_face_short_range.tflite",
        model_options: Dict[str, Any] = None,
    ):
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        RunningMode = mp.tasks.vision.RunningMode
        
        if model_options is None:
            model_options = {}

        # running_mode 처리 (문자열 허용)
        rm = model_options.get("running_mode", "IMAGE")
        if isinstance(rm, str):
            rm = rm.upper()
            if rm == "VIDEO":
                rm = RunningMode.VIDEO
            else:
                rm = RunningMode.IMAGE
        elif rm == RunningMode.LIVE_STREAM:
            # 동기 processor는 LIVE_STREAM 불가 -> IMAGE로 강제
            rm = RunningMode.IMAGE

        opts = dict(model_options)
        opts["running_mode"] = rm

        # 필수 base_options
        opts["base_options"] = BaseOptions(model_asset_path=model_path)

        self._rm = rm
        self._detector = FaceDetector.create_from_options(FaceDetectorOptions(**opts))

    def close(self):
        self._detector.close()

    def process(self, pkt: FramePacket) -> Optional[FramePacket]:
        h, w = pkt.image.shape[:2]
        rgb = cv2.cvtColor(pkt.image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if str(self._rm).endswith("VIDEO"):
            # timestamp는 ms 필요
            result = self._detector.detect_for_video(mp_image, pkt.ts_ms)
        else:
            result = self._detector.detect(mp_image)

        if not result.detections:
            return None

        # 가장 큰 얼굴 1개 선택
        best = None
        best_area = 0
        for det in result.detections:
            bbox = det.bounding_box
            score = det.categories[0].score
            x, y, bw, bh = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            area = bw * bh
            if area > best_area:
                best_area = area
                best = (x, y, bw, bh, float(score))

        if best is None:
            return None

        x, y, bw, bh, score = best
        return FramePacket(
            image=pkt.image,
            ts_ms=pkt.ts_ms,
            seq=pkt.seq,
            source_id=pkt.source_id,
            roi=(x, y, bw, bh),
            meta={**pkt.meta, "face_score": score},
        )
