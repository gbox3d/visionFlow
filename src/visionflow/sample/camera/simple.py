"""
file : sample/camera/simple.py

VisionFlow Standard Renderer Template

Rules:
- This file NEVER performs inference.
- This file NEVER blocks on processing.
- This file ONLY renders what exists on the TopicBus.
- This file is SAFE to copy for other samples.

DO NOT edit this block.
"""

from __future__ import annotations

import argparse
import time
import os
import threading

import pygame

from visionflow.pipeline.bus import TopicBus
from visionflow.sources.camera_source import CameraSource
from visionflow.utils.image import cv_bgr_to_pygame_surface


class Application:
    """
    VisionFlow Standard Renderer
    - Main thread: pygame rendering ONLY
    - Camera runs in its own thread
    """

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        max_fail: int = 60,
        use_dshow: bool = True,
    ):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.max_fail = max_fail
        self.use_dshow = use_dshow

        # Shared bus
        self.bus = TopicBus()

        # Camera source
        self.camera = CameraSource(
            bus=self.bus,
            out_topic="frame/raw",
            camera_id=self.camera_id,
            request_width=self.width,
            request_height=self.height,
            max_fail=self.max_fail,
            use_dshow=self.use_dshow,
            source_id=f"camera{self.camera_id}",
        )

        # pygame setup (render only)
        pygame.init()
        pygame.display.set_caption("VisionFlow - Camera Viewer")

        self.font = pygame.font.SysFont("Consolas", 18)
        self.small = pygame.font.SysFont("Consolas", 14)

        # process / thread info (debug)
        self.main_pid = os.getpid()
        self.main_tid = threading.get_ident()

    def run(self):
        # start camera thread
        self.camera.start()

        # --- wait first frame ---
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
            self.camera.stop()
            raise RuntimeError("Camera frame not received (timeout 5s)")

        # init window size from first frame
        h0, w0 = last_pkt.image.shape[:2]
        screen = pygame.display.set_mode((w0, h0), pygame.RESIZABLE)

        running = True
        last_surface = None

        # display fps
        disp_count = 0
        disp_fps = 0.0
        disp_t0 = time.time()

        while running:
            # --- events ---
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    running = False
                elif e.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode((e.w, e.h), pygame.RESIZABLE)

            # --- get latest frame (non-blocking) ---
            pkt, new_ver = self.bus.wait_latest(
                "frame/raw", last_ver, timeout=0.0
            )
            if pkt is not None:
                last_ver = new_ver
                last_pkt = pkt
                try:
                    last_surface = cv_bgr_to_pygame_surface(pkt.image)
                except Exception:
                    last_surface = None

            if last_pkt is None or last_surface is None:
                time.sleep(0.005)
                continue

            # --- render frame ---
            sw, sh = screen.get_size()
            scaled = pygame.transform.scale(last_surface, (sw, sh))
            screen.blit(scaled, (0, 0))

            # --- fps calc ---
            disp_count += 1
            now = time.time()
            if now - disp_t0 >= 1.0:
                disp_fps = disp_count / (now - disp_t0)
                disp_count = 0
                disp_t0 = now

            meta = last_pkt.meta or {}
            cam_fps = float(meta.get("cam_fps", 0.0))

            line1 = f"VisionFlow Renderer"
            line2 = f"CAM {meta.get('camera_id','?')}  {meta.get('actual_width',0)}x{meta.get('actual_height',0)}"
            line3 = f"disp_fps / cam_fps = {disp_fps:.1f} / {cam_fps:.1f}"

            # overlay background
            pad = 6
            box_w = 420
            box_h = 70
            bg = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 140))
            screen.blit(bg, (10, 10))

            screen.blit(self.font.render(line1, True, (255, 255, 0)), (10 + pad, 10))
            screen.blit(self.small.render(line2, True, (255, 255, 255)), (10 + pad, 10 + 24))
            screen.blit(self.small.render(line3, True, (255, 255, 255)), (10 + pad, 10 + 44))

            pygame.display.flip()
            time.sleep(0.001)

        self.camera.stop()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description="VisionFlow standard camera renderer"
    )
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max-fail", type=int, default=60)
    parser.add_argument("--dshow", type=int, default=1)

    args = parser.parse_args()

    app = Application(
        camera_id=args.camera_id,
        width=args.width,
        height=args.height,
        max_fail=args.max_fail,
        use_dshow=(args.dshow != 0),
    )
    app.run()


if __name__ == "__main__":
    main()
