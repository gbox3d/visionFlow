from __future__ import annotations

import time
import threading
from typing import Any, Dict, Optional

import cv2
import mediapipe as mp

from visionflow.pipeline.bus import TopicBus
from visionflow.pipeline.packet import FramePacket


class MpFaceDetectorAsyncWorker:
    """
    MediaPipe Tasks LIVE_STREAM FaceDetector 전용 워커
    - in_topic: frame/raw
    - out_topic: frame/face
    - detect_async() 호출
    - 결과는 callback에서 bus.publish()

    카메라 / 추론 / 렌더링 완전 분리 유지
    """

    def __init__(
        self,
        bus: TopicBus,
        in_topic: str,
        out_topic: str,
        model_path: str = 'models/blaze_face_short_range.tflite',
        model_options: Dict[str, Any] = None,
        max_faces: int = 1,
        name: str = "mp-face-live",
    ):
        self.bus = bus
        self.in_topic = in_topic
        self.out_topic = out_topic
        self.model_path = model_path
        self.model_options = dict(model_options) if model_options is not None else {}
        self.max_faces = max(1, int(max_faces))
        self.name = name

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_ver = 0

        # infer fps
        self._fps_t0 = time.time()
        self._fps_n = 0
        self._infer_fps = 0.0

        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        RunningMode = mp.tasks.vision.RunningMode

        def _callback(result, output_image, timestamp_ms: int):
            # infer fps
            self._fps_n += 1
            now = time.time()
            d = now - self._fps_t0
            if d >= 1.0:
                self._infer_fps = self._fps_n / d
                self._fps_n = 0
                self._fps_t0 = now

            if result is None or not result.detections:
                return

            # output_image는 mp.Image, 원본 ndarray가 없을 수 있어서
            # 우리는 "latest raw frame"을 다시 가져와서 매칭합니다.
            raw = self.bus.get_latest(self.in_topic)
            if raw is None:
                return

            faces: list[tuple[int, int, int, int, float, int]] = []
            for det in result.detections:
                bbox = det.bounding_box
                score = det.categories[0].score
                x, y, bw, bh = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                area = bw * bh
                faces.append((x, y, bw, bh, float(score), int(area)))

            if not faces:
                return

            faces.sort(key=lambda item: item[5], reverse=True)
            selected = faces[: self.max_faces]
            x, y, bw, bh, score, _ = selected[0]
            face_rois = [(fx, fy, fw, fh, fs) for fx, fy, fw, fh, fs, _ in selected]
            out_pkt = FramePacket(
                image=raw.image,
                ts_ms=raw.ts_ms,
                seq=raw.seq,
                source_id=raw.source_id,
                roi=(x, y, bw, bh),
                meta={
                    **raw.meta,
                    "face_score": score,
                    "face_count": len(face_rois),
                    "face_rois": face_rois,
                    "infer_fps": self._infer_fps,
                },
            )
            self.bus.publish(self.out_topic, out_pkt)

        # 옵션 합성
        opts = dict(self.model_options)
        opts["base_options"] = BaseOptions(model_asset_path=self.model_path)
        opts["running_mode"] = RunningMode.LIVE_STREAM
        opts["result_callback"] = _callback

        self._detector = FaceDetector.create_from_options(FaceDetectorOptions(**opts))

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
        self._detector.close()

    def _loop(self):
        while self._running:
            pkt, v = self.bus.wait_latest(self.in_topic, self._last_ver, timeout=0.2)
            if pkt is None:
                continue
            self._last_ver = v

            rgb = cv2.cvtColor(pkt.image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # LIVE_STREAM은 async + timestamp 필요
            self._detector.detect_async(mp_image, pkt.ts_ms)
            time.sleep(0.001)
