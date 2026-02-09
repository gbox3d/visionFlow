"""
file : sample/camera/dual_cam.py

description:
    두 대의 카메라를 동시에 보여주는 애플리케이션

do not edit this comment block.
"""

from __future__ import annotations

import time
import pygame

from visionflow.pipeline.bus import TopicBus
from visionflow.sources.camera_source import CameraSource
from visionflow.utils.image import cv_bgr_to_pygame_surface


class Application:
    """
    VisionFlow Dual Camera Renderer
    - Camera / Render 완전 분리
    - TopicBus 버전 기반 구독
    """

    def __init__(self):
        # 하나의 Bus를 공유해야 한 화면에 그릴 수 있음
        self.bus = TopicBus()

        self.cam1 = CameraSource(
            bus=self.bus,
            out_topic="frame/raw1",
            camera_id=0,
            request_width=640,
            request_height=480,
            source_id="camera0",
        )
        self.cam2 = CameraSource(
            bus=self.bus,
            out_topic="frame/raw2",
            camera_id=1,
            request_width=640,
            request_height=480,
            source_id="camera1",
        )

        # 버전 관리 (핵심)
        self.ver1 = 0
        self.ver2 = 0

        # 렌더링 FPS 계산용
        self.disp1_cnt = 0
        self.disp1_fps = 0.0
        self.disp1_t0 = time.time()

        self.disp2_cnt = 0
        self.disp2_fps = 0.0
        self.disp2_t0 = time.time()

        self.running = True

    def run(self):
        self.cam1.start()
        self.cam2.start()

        pygame.init()
        pygame.display.set_caption("VisionFlow - Dual Camera Viewer (640x480 x 2)")

        font = pygame.font.SysFont("Consolas", 18)
        small = pygame.font.SysFont("Consolas", 14)

        screen = pygame.display.set_mode((1280, 480), pygame.RESIZABLE)

        last_pkt1 = None
        last_pkt2 = None
        surf1 = None
        surf2 = None

        # 초기 버전 확보
        self.ver1 = self.bus.get_version("frame/raw1")
        self.ver2 = self.bus.get_version("frame/raw2")

        while self.running:
            # ---------- 이벤트 ----------
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.running = False
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    self.running = False
                elif e.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode((e.w, e.h), pygame.RESIZABLE)

            # ---------- 카메라 1 ----------
            pkt1, new_ver1 = self.bus.wait_latest(
                "frame/raw1", self.ver1, timeout=0.0
            )
            if pkt1 is not None:
                self.ver1 = new_ver1
                last_pkt1 = pkt1
                try:
                    surf1 = cv_bgr_to_pygame_surface(pkt1.image)
                except Exception:
                    surf1 = None

            # ---------- 카메라 2 ----------
            pkt2, new_ver2 = self.bus.wait_latest(
                "frame/raw2", self.ver2, timeout=0.0
            )
            if pkt2 is not None:
                self.ver2 = new_ver2
                last_pkt2 = pkt2
                try:
                    surf2 = cv_bgr_to_pygame_surface(pkt2.image)
                except Exception:
                    surf2 = None

            sw, sh = screen.get_size()
            w_half = sw // 2

            screen.fill((0, 0, 0))

            # ---------- 좌측 ----------
            if surf1 is not None:
                screen.blit(
                    pygame.transform.scale(surf1, (w_half, sh)), # 좌우 반씩
                    (0, 0),
                )
                self._update_disp_fps(1)

            # ---------- 우측 ----------
            if surf2 is not None:
                screen.blit(
                    pygame.transform.scale(surf2, (sw - w_half, sh)), # 좌우 반씩
                    (w_half, 0),
                )
                self._update_disp_fps(2)

            # ---------- 오버레이 ----------
            self._draw_overlay(
                screen, font, small,
                x=10, y=10,
                title="CAM0",
                pkt=last_pkt1,
                disp_fps=self.disp1_fps,
            )
            self._draw_overlay(
                screen, font, small,
                x=w_half + 10, y=10,
                title="CAM1",
                pkt=last_pkt2,
                disp_fps=self.disp2_fps,
            )

            pygame.display.flip()
            time.sleep(0.001)

        self.cam1.stop()
        self.cam2.stop()
        pygame.quit()

    # -------------------------------------------------

    def _update_disp_fps(self, idx: int):
        now = time.time()
        if idx == 1:
            self.disp1_cnt += 1
            if now - self.disp1_t0 >= 1.0:
                self.disp1_fps = self.disp1_cnt / (now - self.disp1_t0)
                self.disp1_cnt = 0
                self.disp1_t0 = now
        else:
            self.disp2_cnt += 1
            if now - self.disp2_t0 >= 1.0:
                self.disp2_fps = self.disp2_cnt / (now - self.disp2_t0)
                self.disp2_cnt = 0
                self.disp2_t0 = now

    def _draw_overlay(self, screen, font, small, x, y, title, pkt, disp_fps):
        pad = 6
        box_w = 380
        box_h = 70

        bg = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 140))
        screen.blit(bg, (x, y))

        if pkt is None:
            screen.blit(
                font.render(f"{title}: no frame", True, (255, 120, 120)),
                (x + pad, y + pad),
            )
            return

        meta = pkt.meta or {}
        cam_fps = float(meta.get("cam_fps", 0.0))

        line1 = f"{title}  SRC:{pkt.source_id}"
        line2 = f"SEQ:{pkt.seq}"
        line3 = f"disp_fps/cam_fps = {disp_fps:.1f}/{cam_fps:.1f}"

        screen.blit(font.render(line1, True, (255, 255, 0)), (x + pad, y + 0))
        screen.blit(small.render(line2, True, (255, 255, 255)), (x + pad, y + 24))
        screen.blit(small.render(line3, True, (180, 255, 180)), (x + pad, y + 44))


def main():
    Application().run()


if __name__ == "__main__":
    main()
