# file : sample/pose/simple.py

from __future__ import annotations

import argparse
import time
import os
import threading

import pygame

from visionflow.pipeline.bus import TopicBus
from visionflow.sources.camera_source import CameraSource
from visionflow.utils.image import cv_bgr_to_pygame_surface

from visionflow.processors.pose_landmarker import PoseLandmarkerProcessor
from visionflow.workers.sync_inference_worker import SyncInferenceWorker
from visionflow.workers.mp_pose_landmarker_async_worker import MpPoseLandmarkerAsyncWorker


# Pose Landmark 연결 테이블 정의 (MediaPipe Tasks용)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (16, 18),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (24, 26),
    (25, 27), (26, 28),
    (27, 29), (28, 30),
    (29, 31), (30, 32),
]

class Application:
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

        self.running_mode = args.running_mode.upper()

        model_options = {
            "running_mode": self.running_mode,
            "num_poses": args.num_poses,
            "min_pose_detection_confidence": args.min_pose_detection_confidence,
            "min_pose_presence_confidence": args.min_pose_presence_confidence,
            "min_tracking_confidence": args.min_tracking_confidence,
            "output_segmentation_masks": args.output_segmentation_masks,
        }

        if self.running_mode == "LIVE_STREAM":
            self.worker = MpPoseLandmarkerAsyncWorker(
                bus=self.bus,
                in_topic="frame/raw",
                out_topic="frame/pose",
                model_options=model_options,
            )
        else:
            processor = PoseLandmarkerProcessor(
                model_options=model_options
            )
            self.worker = SyncInferenceWorker(
                bus=self.bus,
                in_topic="frame/raw",
                out_topic="frame/pose",
                processor=processor,
            )

        pygame.init()
        pygame.display.set_caption("VisionFlow - Pose Landmarker")

        self.font = pygame.font.SysFont("Consolas", 18)
        self.small = pygame.font.SysFont("Consolas", 14)

        self.main_pid = os.getpid()
        self.main_tid = threading.get_ident()

    def run(self):
        self.camera.start()
        self.worker.start()

        last_ver = self.bus.get_version("frame/raw")
        last_pkt = None
        t0 = time.time()

        while last_pkt is None and (time.time() - t0) < 5.0:
            pkt, last_ver = self.bus.wait_latest(
                "frame/raw", last_ver, timeout=0.2
            )
            if pkt is not None:
                last_pkt = pkt

        if last_pkt is None:
            self.worker.stop()
            self.camera.stop()
            raise RuntimeError("Camera frame not received (timeout 5s)")

        h0, w0 = last_pkt.image.shape[:2]
        screen = pygame.display.set_mode((w0, h0), pygame.RESIZABLE)

        last_surface = None
        running = True

        disp_n = 0
        disp_fps = 0.0
        disp_t0 = time.time()

        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    running = False
                elif e.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode(
                        (e.w, e.h), pygame.RESIZABLE
                    )

            raw = self.bus.get_latest("frame/raw")
            pose = self.bus.get_latest("frame/pose")

            if raw is None:
                time.sleep(0.005)
                continue

            try:
                last_surface = cv_bgr_to_pygame_surface(raw.image)
            except Exception:
                last_surface = None

            if last_surface is None:
                continue

            sw, sh = screen.get_size()
            scaled = pygame.transform.scale(last_surface, (sw, sh))
            screen.blit(scaled, (0, 0))

            # draw pose landmarks
            if pose is not None:
                lm_list = pose.meta.get("pose_landmarks", [])
                if lm_list:
                    landmarks = lm_list[0]
                    for a, b in POSE_CONNECTIONS:
                        pa = landmarks[a]
                        pb = landmarks[b]
                        ax, ay = int(pa.x * sw), int(pa.y * sh)
                        bx, by = int(pb.x * sw), int(pb.y * sh)
                        pygame.draw.line(screen, (0, 255, 0), (ax, ay), (bx, by), 2)

                    for lm in landmarks:
                        x = int(lm.x * sw)
                        y = int(lm.y * sh)
                        pygame.draw.circle(screen, (255, 0, 0), (x, y), 4)

            disp_n += 1
            now = time.time()
            if now - disp_t0 >= 1.0:
                disp_fps = disp_n / (now - disp_t0)
                disp_n = 0
                disp_t0 = now

            cam_fps = float(raw.meta.get("cam_fps", 0.0))
            infer_fps = 0.0
            if pose is not None:
                infer_fps = float(pose.meta.get("infer_fps", 0.0))

            line1 = "VisionFlow Pose Renderer"
            line2 = f"disp / cam / infer = {disp_fps:.1f} / {cam_fps:.1f} / {infer_fps:.1f}"

            pad = 6
            bg = pygame.Surface((520, 56), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 140))
            screen.blit(bg, (10, 10))

            screen.blit(self.font.render(line1, True, (255, 255, 0)), (10 + pad, 10))
            screen.blit(self.small.render(line2, True, (255, 255, 255)), (10 + pad, 10 + 28))

            pygame.display.flip()
            time.sleep(0.001)

        self.worker.stop()
        self.camera.stop()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description="VisionFlow pose landmarker sample"
    )
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max-fail", type=int, default=60)
    parser.add_argument("--dshow", type=int, default=1)
    
    parser.add_argument("--running-mode", type=str, default="IMAGE",
                    choices=["IMAGE", "VIDEO", "LIVE_STREAM"],
                    help="MediaPipe RunningMode")

    parser.add_argument("--num-poses", type=int, default=1)
    parser.add_argument("--min-pose-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-pose-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--output-segmentation-masks", action="store_true")
    
    
    args = parser.parse_args()
    Application(args).run()

if __name__ == "__main__":
    main()
