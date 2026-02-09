# workers/mp_pose_landmarker_async_worker.py
from __future__ import annotations

import time
import threading
from typing import Any, Dict, Optional

import cv2
import mediapipe as mp

from visionflow.pipeline.bus import TopicBus
from visionflow.pipeline.packet import FramePacket


class MpPoseLandmarkerAsyncWorker:
    """
    MediaPipe Tasks LIVE_STREAM PoseLandmarker 전용 워커

    - in_topic  : frame/raw
    - out_topic : frame/pose
    - detect_async() 사용
    - 결과는 callback에서 bus.publish()

    FaceDetector LIVE_STREAM 워커와 구조 100% 동일
    """

    def __init__(
        self,
        bus: TopicBus,
        in_topic: str,
        out_topic: str,
        model_path: str = "models/pose_landmarker_lite.task",
        model_options: Dict[str, Any] | None = None,
        name: str = "mp-pose-live",
    ):
        self.bus = bus
        self.in_topic = in_topic
        self.out_topic = out_topic
        self.model_path = model_path
        self.model_options = dict(model_options) if model_options else {}
        self.name = name

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_ver = 0

        # infer fps
        self._fps_t0 = time.time()
        self._fps_n = 0
        self._infer_fps = 0.0

        # MediaPipe aliases
        BaseOptions = mp.tasks.BaseOptions
        RunningMode = mp.tasks.vision.RunningMode
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

        # callback 정의
        def _callback(result, output_image, timestamp_ms: int):
            # infer fps 계산
            self._fps_n += 1
            now = time.time()
            d = now - self._fps_t0
            if d >= 1.0:
                self._infer_fps = self._fps_n / d
                self._fps_n = 0
                self._fps_t0 = now

            if result is None or not result.pose_landmarks:
                return

            raw = self.bus.get_latest(self.in_topic)
            if raw is None:
                return

            pkt = FramePacket(
                image=raw.image,
                ts_ms=raw.ts_ms,
                seq=raw.seq,
                source_id=raw.source_id,
                roi=None,
                meta={
                    **raw.meta,
                    "pose_landmarks": result.pose_landmarks,
                    "pose_world_landmarks": result.pose_world_landmarks,
                    "infer_fps": self._infer_fps,
                },
            )
            self.bus.publish(self.out_topic, pkt)

        # 옵션 합성
        opts = dict(self.model_options)
        opts["base_options"] = BaseOptions(model_asset_path=self.model_path)
        opts["running_mode"] = RunningMode.LIVE_STREAM
        opts["result_callback"] = _callback

        self._landmarker = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(**opts)
        )

    # -------------------------------------------------

    def start(self):
        if self._running:
            return
        self._running = True
        self._last_ver = self.bus.get_version(self.in_topic)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._landmarker.close()

    # -------------------------------------------------

    def _loop(self):
        while self._running:
            pkt, v = self.bus.wait_latest(
                self.in_topic, self._last_ver, timeout=0.2
            )
            if pkt is None:
                continue

            self._last_ver = v

            rgb = cv2.cvtColor(pkt.image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            # LIVE_STREAM → async + timestamp 필수
            self._landmarker.detect_async(mp_image, pkt.ts_ms)
            time.sleep(0.001)
