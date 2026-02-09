"""
file : visionFlow/processors/pose_landmarker.py
authors : VisionFlow 개발팀
desc : MediaPipe PoseLandmarker processor (동기)
do not edit this commented blocks
"""


from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import mediapipe as mp

from visionflow.pipeline.packet import FramePacket


class PoseLandmarkerProcessor:
    """
    MediaPipe Tasks API 기반 PoseLandmarker (동기)
    - running_mode: VIDEO 권장
    - process(pkt) -> FramePacket (pose landmarks 포함)
    """

    def __init__(
        self,
        model_path: str = "models/pose_landmarker_lite.task",
        model_options: Dict[str, Any] | None = None,
    ):
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        if model_options is None:
            model_options = {}

        # running_mode 처리 (Face와 동일)
        rm = model_options.get("running_mode", "VIDEO")
        if isinstance(rm, str):
            rm = rm.upper()
            if rm == "IMAGE":
                rm = RunningMode.IMAGE
            else:
                rm = RunningMode.VIDEO
        elif rm == RunningMode.LIVE_STREAM:
            # 동기 processor에서는 LIVE_STREAM 금지
            rm = RunningMode.VIDEO

        opts = dict(model_options)
        opts["running_mode"] = rm
        opts["base_options"] = BaseOptions(model_asset_path=model_path)

        # Pose 옵션
        opts.setdefault("num_poses", 1)
        opts.setdefault("min_pose_detection_confidence", 0.5)
        opts.setdefault("min_pose_presence_confidence", 0.5)
        opts.setdefault("min_tracking_confidence", 0.5)
        opts.setdefault("output_segmentation_masks", False)

        self._rm = rm
        self._landmarker = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(**opts)
        )

    def close(self):
        self._landmarker.close()

    def process(self, pkt: FramePacket) -> Optional[FramePacket]:
        rgb = cv2.cvtColor(pkt.image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if str(self._rm).endswith("VIDEO"):
            result = self._landmarker.detect_for_video(mp_image, pkt.ts_ms)
        else:
            result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        return FramePacket(
            image=pkt.image,
            ts_ms=pkt.ts_ms,
            seq=pkt.seq,
            source_id=pkt.source_id,
            roi=None,
            meta={
                **pkt.meta,
                "pose_landmarks": result.pose_landmarks,
                "pose_world_landmarks": result.pose_world_landmarks,
                "has_segmentation": bool(result.segmentation_masks),
            },
        )
