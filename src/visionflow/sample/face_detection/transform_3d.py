"""
file: sample/face_detection/transform_3d.py
author : VisionFlow 개발팀
date: 2026-02-03

설명:
  - Face Detector -> Face Landmarker 파이프라인 구동
  - Transformation Matrix를 이용해 거리(Distance)와 회전(Euler Angle) 계산
  - 얼굴 중심에 3D 축(Axis) 시각화
  
  
"""

from __future__ import annotations

import argparse
import os
import threading
import time
import math
import numpy as np
import pygame
import cv2

from visionflow.pipeline.bus import TopicBus
from visionflow.sources.camera_source import CameraSource
from visionflow.utils.image import cv_bgr_to_pygame_surface

from visionflow.workers.mp_face_detector_async_worker import MpFaceDetectorAsyncWorker
from visionflow.workers.mp_face_landmarker_async_worker import MpFaceLandmarkerAsyncWorker

from visionflow.utils.face_geometry import FaceGeometryUtils

class Application:
    def __init__(self, args):
        self.args = args
        self.bus = TopicBus()

        # 1. Camera
        self.camera = CameraSource(
            bus=self.bus,
            out_topic="frame/raw",
            camera_id=args.camera_id,
            request_width=args.width,
            request_height=args.height,
            max_fail=args.max_fail,
            use_dshow=bool(args.dshow),
        )

        # 2. Face Detector (ROI 추출용)
        self.detector = MpFaceDetectorAsyncWorker(
            bus=self.bus,
            in_topic="frame/raw",
            out_topic="frame/face",
            model_path="models/blaze_face_short_range.tflite",
            model_options={"min_detection_confidence": 0.5},
        )

        # 3. Face Landmarker (Matrix 추출용)
        # 중요: Worker 내부에서 output_facial_transformation_matrixes=True로 설정되어 있어야 함
        self.landmarker = MpFaceLandmarkerAsyncWorker(
            bus=self.bus,
            in_topic="frame/face",
            out_topic="frame/landmark",
            model_path="models/face_landmarker.task",
            use_roi_crop=True,
            roi_padding=0.3,
            model_options={
                "output_facial_transformation_matrixes": True
            }
                
        )

        pygame.init()
        pygame.display.set_caption("VisionFlow - 3D Face Transform")
        self.font = pygame.font.SysFont("Consolas", 18)
        self.big_font = pygame.font.SysFont("Consolas", 24, bold=True)

    def run(self):
        self.camera.start()
        self.detector.start()
        self.landmarker.start()

        # 첫 프레임 대기
        self._wait_for_camera()
        
        # 화면 설정
        raw = self.bus.get_latest("frame/raw")
        h, w = raw.image.shape[:2]
        screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)

        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    running = False

            # 데이터 가져오기
            raw = self.bus.get_latest("frame/raw")
            landmark_pkt = self.bus.get_latest("frame/landmark")

            if raw is None:
                time.sleep(0.01)
                continue

            # 1. 배경 그리기
            surface = cv_bgr_to_pygame_surface(raw.image)
            sw, sh = screen.get_size()
            scaled_surf = pygame.transform.scale(surface, (sw, sh))
            screen.blit(scaled_surf, (0, 0))

            # 2. 3D 정보 시각화
            if landmark_pkt and landmark_pkt.meta.get("face_matrix") is not None:
                matrix = landmark_pkt.meta["face_matrix"] # 4x4 numpy array
                
                # [A] 거리 계산 (Translation Vector Z값)
                # matrix 구조: [R | T] -> T_x, T_y, T_z는 4열(index 3)
                t_x = matrix[0, 3]
                t_y = matrix[1, 3]
                # t_z = matrix[2, 3] # 거리 (cm)
                
                # [B] 각도 계산 (Rotation Matrix -> Euler)
                pitch, yaw, roll = FaceGeometryUtils.matrix_to_euler_angles(matrix)
                
                dist_cm = 0.0
                if landmark_pkt.meta.get("face_landmarks"):
                    dist_cm = FaceGeometryUtils.estimate_distance_from_eyes(
                        landmarks=landmark_pkt.meta["face_landmarks"][0],
                        image_width=w,
                        image_height=h,
                        yaw_deg=yaw,
                        # [튜닝 포인트] 50cm 거리에서 실제 거리와 맞게 이 값을 조절하세요. 화면 값이 작게 나오면? -> 상수를 키우세요.
                        focal_length_constant=650.0 # 조정 필요
                    )

                # [C] 화면 렌더링
                self._draw_info_panel(screen, t_x, t_y, dist_cm, pitch, yaw, roll)
                
                # [D] 얼굴 중심축 그리기 (간략화)
                # 얼굴 랜드마크가 있으면 코 끝(1번)이나 미간(168번) 좌표를 찾아 그림
                if landmark_pkt.meta.get("face_landmarks"):
                    lms = landmark_pkt.meta["face_landmarks"][0] # 첫번째 얼굴
                    nose_tip = lms[1] # 코 끝
                    
                    # 정규 좌표 -> 화면 좌표
                    nx = int(nose_tip["x"] * sw)
                    ny = int(nose_tip["y"] * sh)
                    
                    self._draw_axis(screen, nx, ny, pitch, yaw, roll)

            pygame.display.flip()
            time.sleep(0.001)

        self.landmarker.stop()
        self.detector.stop()
        self.camera.stop()
        pygame.quit()

    def _wait_for_camera(self):
        t0 = time.time()
        while (time.time() - t0) < 5.0:
            if self.bus.get_latest("frame/raw") is not None:
                return
            time.sleep(0.1)
        raise RuntimeError("Camera Timeout")

    def _draw_info_panel(self, screen, x, y, z, p, y_deg, r):
        """좌측 상단에 수치 정보 표시"""
        # 패널 배경
        bg = pygame.Surface((300, 160), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 180))
        screen.blit(bg, (10, 10))

        # 텍스트 색상 (거리에 따라 변경: 너무 가까우면 빨강)
        dist_color = (0, 255, 0) if z > 30 else (255, 50, 50)
        
        texts = [
            (f"DIST : {abs(z):.1f} cm", dist_color, self.big_font),
            (f"POS  : X={x:.1f}, Y={y:.1f}", (200, 200, 200), self.font),
            (f"----------------", (100, 100, 100), self.font),
            (f"PITCH: {p:.1f}° (UD)", (255, 255, 0), self.font),
            (f"YAW  : {y_deg:.1f}° (LR)", (255, 255, 0), self.font),
            (f"ROLL : {r:.1f}° (Tilt)", (255, 255, 0), self.font),
        ]

        curr_y = 20
        for txt, color, font in texts:
            img = font.render(txt, True, color)
            screen.blit(img, (20, curr_y))
            curr_y += 25

    def _draw_axis(self, screen, cx, cy, pitch, yaw, roll, size=50):
        """
        얼굴 중심(cx, cy)에 2D 투영된 3축 그리기 (매우 단순화된 버전)
        실제 3D 투영 행렬 없이 각도만으로 근사하여 그림
        """
        # Radian 변환
        pr, yr, rr = np.radians([pitch, yaw, roll])

        # Yaw(Y축 회전) -> X축(빨강)의 끝점 이동
        x_end = (cx + size * math.cos(yr), cy + size * math.sin(yr) * math.sin(pr))
        
        # Pitch(X축 회전) -> Y축(초록)의 끝점 이동
        y_end = (cx - size * math.sin(rr), cy - size * math.cos(rr))
        
        # Z축(파랑)은 정면(Yaw, Pitch에 따라 단축됨) - 시각적 효과만
        z_end = (cx, cy) # Z축은 화면 뚫고 나오는 방향이라 점으로 표현됨

        # 축 그리기 (Red: X-좌우, Green: Y-상하, Blue: Z-앞뒤)
        pygame.draw.line(screen, (255, 0, 0), (cx, cy), x_end, 3) # X axis
        pygame.draw.line(screen, (0, 255, 0), (cx, cy), y_end, 3) # Y axis
        pygame.draw.circle(screen, (0, 0, 255), (cx, cy), 5)      # Center (Z origin)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max-fail", type=int, default=60)
    parser.add_argument("--dshow", type=int, default=1)
    args = parser.parse_args()

    Application(args).run()

if __name__ == "__main__":
    main()