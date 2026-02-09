"""
file : sample/face_detection/simple.py

important:
dont not edit this commented block
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

from visionflow.processors.face_detector import FaceDetectorProcessor
from visionflow.workers.sync_inference_worker import SyncInferenceWorker
from visionflow.workers.mp_face_detector_async_worker import MpFaceDetectorAsyncWorker


class Application:
    """
    - Main Thread: pygame 렌더링만 담당
    - Camera Thread: 캡처만 담당
    - Inference Thread: 추론만 담당 (IMAGE/VIDEO) 또는 LIVE_STREAM Async Worker
    """

    def __init__(self, args):
        self.args = args
        self.bus = TopicBus()

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

        # model options (공통)
        model_options = {
            "min_detection_confidence": float(args.min_score),
            
        }
        if args.min_suppression_threshold is not None:
            model_options["min_suppression_threshold"] = float(args.min_suppression_threshold)

        # running mode
        self.running_mode = args.running_mode.upper()

        if self.running_mode == "LIVE_STREAM":
            self.worker = MpFaceDetectorAsyncWorker(
                bus=self.bus,
                in_topic="frame/raw",
                out_topic="frame/face",
                # model_path=args.model_path,
                model_options=model_options,
            )
        else:
            model_options["running_mode"] = self.running_mode  # IMAGE or VIDEO
            processor = FaceDetectorProcessor(
                # model_path=args.model_path,
                model_options=model_options,
            )
            self.worker = SyncInferenceWorker(
                bus=self.bus,
                in_topic="frame/raw",
                out_topic="frame/face",
                processor=processor,
            )

        pygame.init()
        pygame.display.set_caption("VisionFlow - Face Detection")

        self.font = pygame.font.SysFont("Consolas", 18)
        self.small = pygame.font.SysFont("Consolas", 14)

        self.main_pid = os.getpid()
        self.main_tid = threading.get_ident()

    def run(self):
        self.camera.start()
        self.worker.start()

        # 첫 프레임 대기
        t0 = time.time()
        raw = None
        while raw is None and (time.time() - t0) < 5.0:
            raw = self.bus.get_latest("frame/raw")
            time.sleep(0.05)

        if raw is None:
            self.worker.stop()
            self.camera.stop()
            raise RuntimeError("카메라 프레임 수신 실패 (5초 동안 프레임 없음)")

        h0, w0 = raw.image.shape[:2]
        screen = pygame.display.set_mode((w0, h0), pygame.RESIZABLE)

        # disp fps
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
            face = self.bus.get_latest("frame/face")  # roi 포함된 최신 결과

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

            # bbox overlay (face roi가 있으면)
            if face is not None and face.roi is not None:
                x, y, w, h = face.roi
                # 화면이 리사이즈 되었으니 좌표 스케일 보정
                sx = sw / raw.image.shape[1]
                sy = sh / raw.image.shape[0]
                rx = int(x * sx)
                ry = int(y * sy)
                rw = int(w * sx)
                rh = int(h * sy)
                pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(rx, ry, rw, rh), 2)

            # FPS 계산
            disp_n += 1
            now = time.time()
            dt = now - disp_t0
            if dt >= 1.0:
                disp_fps = disp_n / dt
                disp_n = 0
                disp_t0 = now

            cam_fps = float(raw.meta.get("cam_fps", 0.0))
            infer_fps = 0.0
            face_score = 0.0
            if face is not None:
                infer_fps = float(face.meta.get("infer_fps", 0.0))
                face_score = float(face.meta.get("face_score", 0.0))

            line1 = f"MODE: {self.running_mode}   CAM ID: {raw.meta.get('camera_id','?')}   SRC: {raw.source_id}"
            line2 = f"REQ: {raw.meta.get('request_width',0)}x{raw.meta.get('request_height',0)}   ACT: {raw.meta.get('actual_width',0)}x{raw.meta.get('actual_height',0)}"
            line3 = f"SEQ: {raw.seq}   disp_fps/cam_fps/infer_fps = {disp_fps:.1f}/{cam_fps:.1f}/{infer_fps:.1f}   score={face_score:.2f}"

            pad = 6
            box_w = 740
            box_h = 74
            bg = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 140))
            screen.blit(bg, (10, 10))

            screen.blit(self.font.render(line1, True, (255, 255, 0)), (10 + pad, 10 + 0))
            screen.blit(self.small.render(line2, True, (255, 255, 255)), (10 + pad, 10 + 26))
            screen.blit(self.small.render(line3, True, (255, 255, 255)), (10 + pad, 10 + 48))

            pygame.display.flip()
            time.sleep(0.001)

        self.worker.stop()
        self.camera.stop()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="VisionFlow face detection sample (mediapipe tasks)")

    parser.add_argument("--camera-id", type=int, default=0, help="camera index")
    parser.add_argument("--width", type=int, default=640, help="requested width")
    parser.add_argument("--height", type=int, default=480, help="requested height")
    parser.add_argument("--max-fail", type=int, default=60, help="max consecutive read fails before reconnect")
    parser.add_argument("--dshow", type=int, default=1, help="1: use CAP_DSHOW on Windows (recommended)")

    # parser.add_argument("--model-path", type=str, required=True, help="path to face_detector.task")
    parser.add_argument("--running-mode", type=str, default="IMAGE", choices=["IMAGE", "VIDEO", "LIVE_STREAM"], help="MediaPipe RunningMode")
    parser.add_argument("--min-score", type=float, default=0.5, help="min_detection_confidence")
    parser.add_argument("--min-suppression-threshold", type=float, default=None, help="min_suppression_threshold (optional)")

    args = parser.parse_args()
    Application(args).run()


if __name__ == "__main__":
    main()
