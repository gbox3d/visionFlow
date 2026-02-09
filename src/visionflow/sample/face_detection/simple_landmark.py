"""
file: sample/face_detection/simple_landmark.py
author : VisionFlow 개발팀
date: 2026-02-03

얼굴 감지 → ROI → 랜드마크 검출 파이프라인 예제

Pipeline:
  CameraSource → frame/raw
       ↓
  FaceDetectorAsyncWorker → frame/face (ROI 포함)
       ↓
  FaceLandmarkerAsyncWorker → frame/landmark (랜드마크 포함)
       ↓
  Application (렌더링)

각 워커는 완전히 비동기로 동작하며, 큐잉 없이 최신 프레임만 사용


이 주석은 수정하지마세요.
"""

from __future__ import annotations

import argparse
import os
import threading
import time

import pygame

from visionflow.pipeline.bus import TopicBus
from visionflow.sources.camera_source import CameraSource
from visionflow.utils.image import cv_bgr_to_pygame_surface

from visionflow.workers.mp_face_detector_async_worker import MpFaceDetectorAsyncWorker
from visionflow.workers.mp_face_landmarker_async_worker import MpFaceLandmarkerAsyncWorker


# MediaPipe Face Mesh 연결 정보 (얼굴 윤곽선용)
FACE_OVAL_CONNECTIONS = [
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
]

# 눈, 입 등 주요 연결 (간소화)
LEFT_EYE_CONNECTIONS = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (133, 173), (173, 157), (157, 158), (158, 159),
    (159, 160), (160, 161), (161, 246), (246, 33),
]

RIGHT_EYE_CONNECTIONS = [
    (362, 382), (382, 381), (381, 380), (380, 374), (374, 373), (373, 390),
    (390, 249), (249, 263), (263, 466), (466, 388), (388, 387), (387, 386),
    (386, 385), (385, 384), (384, 398), (398, 362),
]

LIPS_CONNECTIONS = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
    (314, 405), (405, 321), (321, 375), (375, 291), (291, 409), (409, 270),
    (270, 269), (269, 267), (267, 0), (0, 37), (37, 39), (39, 40),
    (40, 185), (185, 61),
]


class Application:
    """
    3단계 비동기 파이프라인:
    - Main Thread: pygame 렌더링
    - Camera Thread: 캡처
    - Face Detector Thread: 얼굴 감지 (LIVE_STREAM)
    - Face Landmarker Thread: 랜드마크 검출 (LIVE_STREAM)
    """

    def __init__(self, args):
        self.args = args
        self.bus = TopicBus()

        # 1) Camera Source
        self.camera = CameraSource(
            bus=self.bus,
            out_topic="frame/raw",
            camera_id=args.camera_id,
            request_width=args.width,
            request_height=args.height,
            max_fail=args.max_fail,
            use_dshow=bool(args.dshow),
            source_id=f"camera{args.camera_id}",
        )

        # 2) Face Detector (LIVE_STREAM)
        face_detector_options = {
            "num_faces": int(args.num_faces),
            "min_detection_confidence": float(args.min_detection_confidence),
        }
        self.face_detector = MpFaceDetectorAsyncWorker(
            bus=self.bus,
            in_topic="frame/raw",
            out_topic="frame/face",
            model_path=args.face_model_path,
            model_options=face_detector_options,
            name="face-detector",
        )

        # 3) Face Landmarker (LIVE_STREAM)
        face_landmarker_options = {
            # "num_faces": int(args.num_faces),
            "min_face_detection_confidence": float(args.min_face_detection_confidence),
            "min_face_presence_confidence": float(args.min_face_presence_confidence),
            "min_tracking_confidence": float(args.min_tracking_confidence),
            "output_face_blendshapes": bool(args.output_blendshapes),
        }
        self.face_landmarker = MpFaceLandmarkerAsyncWorker(
            bus=self.bus,
            in_topic="frame/face",
            out_topic="frame/landmark",
            model_path=args.landmark_model_path,
            model_options=face_landmarker_options,
            use_roi_crop=bool(args.use_roi_crop),
            roi_padding=float(args.roi_padding),
            name="face-landmarker",
        )

        pygame.init()
        pygame.display.set_caption("VisionFlow - Face Detection + Landmark")

        self.font = pygame.font.SysFont("Consolas", 18)
        self.small = pygame.font.SysFont("Consolas", 14)

        self.main_pid = os.getpid()
        self.main_tid = threading.get_ident()

    def run(self):
        # 순서대로 시작
        self.camera.start()
        self.face_detector.start()
        self.face_landmarker.start()

        # 첫 프레임 대기
        t0 = time.time()
        raw = None
        while raw is None and (time.time() - t0) < 5.0:
            raw = self.bus.get_latest("frame/raw")
            time.sleep(0.05)

        if raw is None:
            self.face_landmarker.stop()
            self.face_detector.stop()
            self.camera.stop()
            raise RuntimeError("카메라 프레임 수신 실패 (5초 동안 프레임 없음)")

        h0, w0 = raw.image.shape[:2]
        screen = pygame.display.set_mode((w0, h0), pygame.RESIZABLE)

        # 디스플레이 fps
        disp_n = 0
        disp_fps = 0.0
        disp_t0 = time.time()

        running = True
        last_surface = None

        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    running = False
                elif e.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode((e.w, e.h), pygame.RESIZABLE)

            raw = self.bus.get_latest("frame/raw")
            face = self.bus.get_latest("frame/face")
            landmark = self.bus.get_latest("frame/landmark")

            if raw is None:
                time.sleep(0.01)
                continue

            try:
                last_surface = cv_bgr_to_pygame_surface(raw.image)
            except Exception:
                last_surface = None

            if last_surface is None:
                time.sleep(0.01)
                continue

            sw, sh = screen.get_size()
            scaled = pygame.transform.scale(last_surface, (sw, sh))
            screen.blit(scaled, (0, 0))

            img_h, img_w = raw.image.shape[:2]
            sx = sw / img_w
            sy = sh / img_h

            # Face ROI (from face detector)
            if face is not None and face.roi is not None:
                x, y, w, h = face.roi
                rx = int(x * sx)
                ry = int(y * sy)
                rw = int(w * sx)
                rh = int(h * sy)
                pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(rx, ry, rw, rh), 2)

            # Face Landmarks (from face landmarker)
            face_landmarks_data = None
            if landmark is not None:
                face_landmarks_data = landmark.meta.get("face_landmarks")

            if face_landmarks_data is not None and len(face_landmarks_data) > 0:
                for face_lms in face_landmarks_data:
                    # 랜드마크 점 그리기 (일부만 표시)
                    for i, lm in enumerate(face_lms):
                        px = int(lm["x"] * img_w * sx)
                        py = int(lm["y"] * img_h * sy)
                        # 모든 점은 작게
                        pygame.draw.circle(screen, (255, 255, 255), (px, py), 1)

                    # 주요 연결선 그리기
                    self._draw_connections(
                        screen, face_lms, FACE_OVAL_CONNECTIONS,
                        (180, 180, 180), img_w, img_h, sx, sy
                    )
                    self._draw_connections(
                        screen, face_lms, LEFT_EYE_CONNECTIONS,
                        (0, 255, 255), img_w, img_h, sx, sy
                    )
                    self._draw_connections(
                        screen, face_lms, RIGHT_EYE_CONNECTIONS,
                        (0, 255, 255), img_w, img_h, sx, sy
                    )
                    self._draw_connections(
                        screen, face_lms, LIPS_CONNECTIONS,
                        (255, 0, 128), img_w, img_h, sx, sy
                    )

            # FPS 계산
            disp_n += 1
            now = time.time()
            dt = now - disp_t0
            if dt >= 1.0:
                disp_fps = disp_n / dt
                disp_n = 0
                disp_t0 = now

            cam_fps = float(raw.meta.get("cam_fps", 0.0))
            face_infer_fps = 0.0
            landmark_infer_fps = 0.0
            face_score = 0.0

            if face is not None:
                face_infer_fps = float(face.meta.get("infer_fps", 0.0))
                face_score = float(face.meta.get("face_score", 0.0))

            if landmark is not None:
                landmark_infer_fps = float(landmark.meta.get("infer_fps", 0.0))

            # 정보 표시
            line1 = f"CAM ID: {raw.meta.get('camera_id', '?')}   SRC: {raw.source_id}"
            line2 = f"REQ: {raw.meta.get('request_width', 0)}x{raw.meta.get('request_height', 0)}   ACT: {raw.meta.get('actual_width', 0)}x{raw.meta.get('actual_height', 0)}"
            line3 = f"disp/cam/face/landmark fps = {disp_fps:.1f}/{cam_fps:.1f}/{face_infer_fps:.1f}/{landmark_infer_fps:.1f}"
            line4 = f"face_score={face_score:.2f}   use_roi_crop={self.args.use_roi_crop}"

            pad = 6
            box_w = 620
            box_h = 94
            bg = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 140))
            screen.blit(bg, (10, 10))

            screen.blit(self.font.render(line1, True, (255, 255, 0)), (10 + pad, 10 + 0))
            screen.blit(self.small.render(line2, True, (255, 255, 255)), (10 + pad, 10 + 24))
            screen.blit(self.small.render(line3, True, (255, 255, 255)), (10 + pad, 10 + 46))
            screen.blit(self.small.render(line4, True, (255, 255, 255)), (10 + pad, 10 + 68))

            pygame.display.flip()
            time.sleep(0.001)

        # 역순으로 정지
        self.face_landmarker.stop()
        self.face_detector.stop()
        self.camera.stop()
        pygame.quit()

    def _draw_connections(
        self,
        screen,
        landmarks,
        connections,
        color,
        img_w,
        img_h,
        sx,
        sy,
    ):
        """랜드마크 연결선 그리기"""
        for start_idx, end_idx in connections:
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            x1 = int(start["x"] * img_w * sx)
            y1 = int(start["y"] * img_h * sy)
            x2 = int(end["x"] * img_w * sx)
            y2 = int(end["y"] * img_h * sy)
            pygame.draw.line(screen, color, (x1, y1), (x2, y2), 1)


def main():
    parser = argparse.ArgumentParser(
        description="VisionFlow face detection + landmark sample"
    )

    # 카메라 설정
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max-fail", type=int, default=60)
    parser.add_argument("--dshow", type=int, default=1)

    # Face Detector 설정
    parser.add_argument(
        "--face-model-path",
        type=str,
        default="models/blaze_face_short_range.tflite",
        help="Face detector model path",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Face detector min confidence",
    )

    # Face Landmarker 설정
    parser.add_argument(
        "--landmark-model-path",
        type=str,
        default="models/face_landmarker.task",
        help="Face landmarker model path",
    )
    parser.add_argument("--num-faces", type=int, default=1)
    parser.add_argument("--min-face-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-face-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--output-blendshapes", type=int, default=0)

    # ROI crop 설정
    parser.add_argument(
        "--use-roi-crop",
        type=int,
        default=1,
        help="1: crop ROI from face detector, 0: use full image",
    )
    parser.add_argument(
        "--roi-padding",
        type=float,
        default=0.3,
        help="ROI padding ratio",
    )

    args = parser.parse_args()
    Application(args).run()


if __name__ == "__main__":
    main()
