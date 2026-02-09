# workers/mp_face_landmarker_async_worker.py
"""
MediaPipe Tasks LIVE_STREAM FaceLandmarker 전용 워커

- in_topic: frame/face (FaceDetector가 발행한 ROI 포함 패킷)
- out_topic: frame/landmark
- detect_async() 사용
- ROI 영역을 crop하여 랜드마크 검출 (선택 사항)
"""
from __future__ import annotations

import time
import threading
from typing import Any, Dict, Optional

import cv2
import numpy as np
import mediapipe as mp

from visionflow.pipeline.bus import TopicBus
from visionflow.pipeline.packet import FramePacket


class MpFaceLandmarkerAsyncWorker:
    """
    MediaPipe Tasks LIVE_STREAM FaceLandmarker 비동기 워커

    - in_topic에서 프레임 구독 (ROI가 있으면 crop하여 처리)
    - 랜드마크 결과를 out_topic으로 발행
    """

    def __init__(
        self,
        bus: TopicBus,
        in_topic: str,
        out_topic: str,
        model_path: str = "models/face_landmarker.task",
        model_options: Dict[str, Any] | None = None,
        use_roi_crop: bool = True,
        roi_padding: float = 0.2,  # ROI 주변 여백 비율
        name: str = "mp-face-landmark-live",
    ):
        self.bus = bus
        self.in_topic = in_topic
        self.out_topic = out_topic
        self.model_path = model_path
        self.model_options = dict(model_options) if model_options else {}
        self.use_roi_crop = use_roi_crop
        self.roi_padding = roi_padding
        self.name = name

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_ver = 0

        # infer fps
        self._fps_t0 = time.time()
        self._fps_n = 0
        self._infer_fps = 0.0

        # 마지막 처리한 패킷 정보 (callback에서 사용)
        self._last_pkt: Optional[FramePacket] = None
        self._last_crop_info: Optional[Dict] = None
        self._pkt_lock = threading.Lock()

        # 타임스탬프 관리 (LIVE_STREAM은 반드시 단조 증가해야 함)
        self._last_ts_ms = 0

        # MediaPipe aliases
        BaseOptions = mp.tasks.BaseOptions
        RunningMode = mp.tasks.vision.RunningMode
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

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

            if result is None or not result.face_landmarks:
                return

            with self._pkt_lock:
                pkt = self._last_pkt
                crop_info = self._last_crop_info

            if pkt is None:
                return
            
            # 변환 행렬 추출 ---
            # MediaPipe 결과에서 matrix가 있으면 numpy array로 변환하여 메타에 저장
            face_matrix = None
            if hasattr(result, "facial_transformation_matrixes") and result.facial_transformation_matrixes:
                # 첫 번째 얼굴의 매트릭스 (4x4 numpy array)
                face_matrix = np.array(result.facial_transformation_matrixes[0])

            # 랜드마크 좌표를 원본 이미지 좌표로 변환 (crop한 경우)
            face_landmarks = result.face_landmarks
            transformed_landmarks = []

            if crop_info is not None and self.use_roi_crop:
                # crop 영역 정보
                cx, cy = crop_info["crop_x"], crop_info["crop_y"]
                cw, ch = crop_info["crop_w"], crop_info["crop_h"]
                orig_h, orig_w = crop_info["orig_h"], crop_info["orig_w"]

                for face_lms in face_landmarks:
                    transformed_face = []
                    for lm in face_lms:
                        # normalized coords → crop 이미지의 pixel coords
                        px = lm.x * cw + cx
                        py = lm.y * ch + cy
                        # 다시 원본 이미지의 normalized coords로
                        nx = px / orig_w
                        ny = py / orig_h
                        transformed_face.append({
                            "x": nx,
                            "y": ny,
                            "z": lm.z,
                        })
                    transformed_landmarks.append(transformed_face)
            else:
                # crop 안 한 경우 그대로
                for face_lms in face_landmarks:
                    transformed_face = []
                    for lm in face_lms:
                        transformed_face.append({
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z,
                        })
                    transformed_landmarks.append(transformed_face)

            out_pkt = FramePacket(
                image=pkt.image,
                ts_ms=pkt.ts_ms,
                seq=pkt.seq,
                source_id=pkt.source_id,
                roi=pkt.roi,
                meta={
                    **pkt.meta,
                    "face_landmarks": transformed_landmarks,
                    "face_landmarks_raw": result.face_landmarks,
                    "face_matrix": face_matrix,
                    "infer_fps": self._infer_fps,
                },
            )
            self.bus.publish(self.out_topic, out_pkt)

        # 옵션 합성
        opts = dict(self.model_options)
        opts["base_options"] = BaseOptions(model_asset_path=self.model_path)
        opts["running_mode"] = RunningMode.LIVE_STREAM
        opts["result_callback"] = _callback

        self._landmarker = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(**opts)
        )

    def _crop_with_padding(
        self, image: np.ndarray, roi: tuple
    ) -> tuple[np.ndarray, Dict]:
        """
        ROI 영역을 padding과 함께 crop
        반환: (crop된 이미지, crop 정보 dict)
        """
        h, w = image.shape[:2]
        x, y, rw, rh = roi

        # padding 적용
        pad_w = int(rw * self.roi_padding)
        pad_h = int(rh * self.roi_padding)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + rw + pad_w)
        y2 = min(h, y + rh + pad_h)

        cropped = image[y1:y2, x1:x2].copy()

        crop_info = {
            "crop_x": x1,
            "crop_y": y1,
            "crop_w": x2 - x1,
            "crop_h": y2 - y1,
            "orig_w": w,
            "orig_h": h,
        }
        return cropped, crop_info

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

    def _loop(self):
        while self._running:
            
            # 마지막 처리한 패킷 가져오기
            pkt, v = self.bus.wait_latest(
                self.in_topic, self._last_ver, timeout=0.2
            )
            if pkt is None:
                continue

            self._last_ver = v

            # 타임스탬프가 이전보다 작거나 같으면 스킵 (LIVE_STREAM 필수 조건)
            if pkt.ts_ms <= self._last_ts_ms:
                continue
            self._last_ts_ms = pkt.ts_ms

            # ROI가 있고 use_roi_crop이면 crop
            if self.use_roi_crop and pkt.roi is not None:
                cropped, crop_info = self._crop_with_padding(pkt.image, pkt.roi)
                with self._pkt_lock:
                    self._last_pkt = pkt
                    self._last_crop_info = crop_info
                rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            else:
                with self._pkt_lock:
                    self._last_pkt = pkt
                    self._last_crop_info = None
                rgb = cv2.cvtColor(pkt.image, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            # LIVE_STREAM → async + timestamp 필수
            self._landmarker.detect_async(mp_image, pkt.ts_ms)
            time.sleep(0.001)