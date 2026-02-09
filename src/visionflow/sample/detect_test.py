"""
file : sample/detect_test.py
author : VisionFlow 개발팀

간단한 객체 검출 테스트 예제

이 주석은 수정하지마세요.
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

from visionflow.workers.mp_face_detector_async_worker import MpFaceDetectorAsyncWorker
from visionflow.workers.mp_face_landmarker_async_worker import MpFaceLandmarkerAsyncWorker
from visionflow.workers.mp_pose_landmarker_async_worker import MpPoseLandmarkerAsyncWorker

from visionflow.utils.draw_utils import DrawUtils, Colors
from visionflow.utils.etc import resource_path


class Application:
    """
    VisionFlow Standard Detector Tester
    - Main thread: pygame rendering ONLY
    - Camera runs in its own thread
    - face , face/landmark , pose 등 다양한 토픽을 구독하여 렌더링 테스트            
    
    키보드 조작:
      [1] Face Detection 토글
      [2] Face Landmark 토글
      [3] Pose Landmark 토글
      [H] 도움말 토글
      [ESC] 종료
    """

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        max_fail: int = 60,
        use_dshow: bool = True,
        face_model_path: str = "models/blaze_face_short_range.tflite",
        landmark_model_path: str = "models/face_landmarker.task",
        pose_model_path: str = "models/pose_landmarker.task",
    ):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.max_fail = max_fail
        self.use_dshow = use_dshow
        self.face_model_path = resource_path(face_model_path)
        self.landmark_model_path = resource_path(landmark_model_path)
        self.pose_model_path = resource_path(pose_model_path)

        # 기능 토글 상태
        self.enable_face_detection = True
        self.enable_face_landmark = True
        self.enable_pose_landmark = True
        self.show_help = True

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

        # Face Detector Worker
        self.face_detector = MpFaceDetectorAsyncWorker(
            bus=self.bus,
            in_topic="frame/raw",
            out_topic="frame/face",
            model_path=self.face_model_path,
            model_options={
                "min_detection_confidence": 0.5,
            },
            name="face-detector",
        )

        # Face Landmarker Worker
        self.face_landmarker = MpFaceLandmarkerAsyncWorker(
            bus=self.bus,
            in_topic="frame/face",
            out_topic="frame/landmark",
            model_path=self.landmark_model_path,
            model_options={
                "min_face_detection_confidence": 0.5,
                "min_face_presence_confidence": 0.5,
                "min_tracking_confidence": 0.5,
            },
            use_roi_crop=True,
            roi_padding=0.3,
            name="face-landmarker",
        )

        # Pose Landmarker Worker
        self.pose_landmarker = MpPoseLandmarkerAsyncWorker(
            bus=self.bus,
            in_topic="frame/raw",
            out_topic="frame/pose",
            model_options={
                "num_poses": 1,
                "min_pose_detection_confidence": 0.5,
                "min_pose_presence_confidence": 0.5,
                "min_tracking_confidence": 0.5,
            },
            name="pose-landmarker",
        )

        # pygame setup (render only)
        pygame.init()
        pygame.display.set_caption("VisionFlow - All Pipeline Tester")

        # 한글 폰트 설정
        font_path = "font/DungGeunMo.ttf"
        self.font = pygame.font.Font(font_path, 18)
        self.small = pygame.font.Font(font_path, 14)
        self.big_font = pygame.font.Font(font_path, 20)

        # process / thread info (debug)
        self.main_pid = os.getpid()
        self.main_tid = threading.get_ident()

    def run(self):
        # 워커들 시작
        self.camera.start()
        self.face_detector.start()
        self.face_landmarker.start()
        self.pose_landmarker.start()

        # 첫 프레임 대기
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
            self._stop_all_workers()
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
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        running = False
                    elif e.key == pygame.K_1:
                        self.enable_face_detection = not self.enable_face_detection
                    elif e.key == pygame.K_2:
                        self.enable_face_landmark = not self.enable_face_landmark
                    elif e.key == pygame.K_3:
                        self.enable_pose_landmark = not self.enable_pose_landmark
                    elif e.key == pygame.K_4:
                        self._switch_camera()
                    elif e.key == pygame.K_h:
                        self.show_help = not self.show_help
                elif e.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode((e.w, e.h), pygame.RESIZABLE)

            # --- get latest data ---
            raw = self.bus.get_latest("frame/raw")
            face_pkt = self.bus.get_latest("frame/face")
            landmark_pkt = self.bus.get_latest("frame/landmark")
            pose_pkt = self.bus.get_latest("frame/pose")

            if raw is None:
                time.sleep(0.005)
                continue

            try:
                last_surface = cv_bgr_to_pygame_surface(raw.image)
            except Exception:
                last_surface = None

            if last_surface is None:
                time.sleep(0.005)
                continue

            # --- render frame ---
            sw, sh = screen.get_size()
            img_h, img_w = raw.image.shape[:2]
            sx = sw / img_w
            sy = sh / img_h

            scaled = pygame.transform.scale(last_surface, (sw, sh))
            screen.blit(scaled, (0, 0))

            # --- Face Detection ROI 렌더링 ---
            if self.enable_face_detection and face_pkt is not None:
                self._draw_face_roi(screen, face_pkt, sx, sy)

            # --- Face Landmark 렌더링 ---
            if self.enable_face_landmark and landmark_pkt is not None:
                self._draw_face_landmarks(screen, landmark_pkt, img_w, img_h, sx, sy)

            # --- Pose Landmark 렌더링 ---
            if self.enable_pose_landmark and pose_pkt is not None:
                self._draw_pose_landmarks(screen, pose_pkt, sw, sh)

            # --- fps calc ---
            disp_count += 1
            now = time.time()
            if now - disp_t0 >= 1.0:
                disp_fps = disp_count / (now - disp_t0)
                disp_count = 0
                disp_t0 = now

            # --- 정보 패널 렌더링 ---
            self._draw_info_panel(
                screen, raw, face_pkt, landmark_pkt, pose_pkt, disp_fps
            )

            # --- 도움말 렌더링 ---
            if self.show_help:
                self._draw_help_panel(screen, sw, sh)

            pygame.display.flip()
            time.sleep(0.001)

        self._stop_all_workers()
        pygame.quit()

    def _stop_all_workers(self):
        """모든 워커 역순으로 정지"""
        self.pose_landmarker.stop()
        self.face_landmarker.stop()
        self.face_detector.stop()
        self.camera.stop()

    def _switch_camera(self):
        """카메라 디바이스 전환 (0 <-> 1)"""
        # 새 카메라 ID
        new_camera_id = 1 if self.camera_id == 0 else 0

        print(f"[INFO] Switching camera: {self.camera_id} -> {new_camera_id}")

        # 모든 워커 정지
        self._stop_all_workers()

        # 카메라 ID 업데이트
        self.camera_id = new_camera_id

        # 카메라 재생성
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

        # Face Detector 재생성
        self.face_detector = MpFaceDetectorAsyncWorker(
            bus=self.bus,
            in_topic="frame/raw",
            out_topic="frame/face",
            model_path=self.face_model_path,
            model_options={
                "min_detection_confidence": 0.5,
            },
            name="face-detector",
        )

        # Face Landmarker 재생성
        self.face_landmarker = MpFaceLandmarkerAsyncWorker(
            bus=self.bus,
            in_topic="frame/face",
            out_topic="frame/landmark",
            model_path=self.landmark_model_path,
            model_options={
                "min_face_detection_confidence": 0.5,
                "min_face_presence_confidence": 0.5,
                "min_tracking_confidence": 0.5,
            },
            use_roi_crop=True,
            roi_padding=0.3,
            name="face-landmarker",
        )

        # Pose Landmarker 재생성
        self.pose_landmarker = MpPoseLandmarkerAsyncWorker(
            bus=self.bus,
            in_topic="frame/raw",
            out_topic="frame/pose",
            model_options={
                "num_poses": 1,
                "min_pose_detection_confidence": 0.5,
                "min_pose_presence_confidence": 0.5,
                "min_tracking_confidence": 0.5,
            },
            name="pose-landmarker",
        )

        # 다시 시작
        self.camera.start()
        self.face_detector.start()
        self.face_landmarker.start()
        self.pose_landmarker.start()

        print(f"[INFO] Camera switched to {self.camera_id}")

    def _draw_face_roi(self, screen, face_pkt, sx, sy):
        """Face Detection ROI 박스 그리기"""
        if face_pkt.roi is None:
            return

        score = face_pkt.meta.get("face_score", 0.0)
        label = f"Face: {score:.2f}"

        DrawUtils.draw_face_roi(
            screen=screen,
            roi=face_pkt.roi,
            scale_x=sx,
            scale_y=sy,
            color=Colors.GREEN,
            thickness=2,
            label=label,
            font=self.small,
        )

    def _draw_face_landmarks(self, screen, landmark_pkt, img_w, img_h, sx, sy):
        """Face Landmark 그리기"""
        face_landmarks_data = landmark_pkt.meta.get("face_landmarks")
        if face_landmarks_data is None or len(face_landmarks_data) == 0:
            return

        DrawUtils.draw_face_landmarks(
            screen=screen,
            face_landmarks_data=face_landmarks_data,
            img_w=img_w,
            img_h=img_h,
            scale_x=sx,
            scale_y=sy,
        )

    def _draw_pose_landmarks(self, screen, pose_pkt, sw, sh):
        """Pose Landmark 그리기"""
        lm_list = pose_pkt.meta.get("pose_landmarks", [])
        if not lm_list:
            return

        DrawUtils.draw_pose_landmarks(
            screen=screen,
            pose_landmarks_data=lm_list,
            screen_w=sw,
            screen_h=sh,
        )

    def _draw_info_panel(self, screen, raw, face_pkt, landmark_pkt, pose_pkt, disp_fps):
        """좌측 상단 정보 패널"""
        meta = raw.meta or {}
        cam_fps = float(meta.get("cam_fps", 0.0))

        face_fps = 0.0
        landmark_fps = 0.0
        pose_fps = 0.0

        if face_pkt is not None:
            face_fps = float(face_pkt.meta.get("infer_fps", 0.0))
        if landmark_pkt is not None:
            landmark_fps = float(landmark_pkt.meta.get("infer_fps", 0.0))
        if pose_pkt is not None:
            pose_fps = float(pose_pkt.meta.get("infer_fps", 0.0))

        lines = [
            f"VisionFlow All Pipeline Tester",
            f"CAM {meta.get('camera_id', '?')}  {meta.get('actual_width', 0)}x{meta.get('actual_height', 0)}",
            f"disp/cam = {disp_fps:.1f} / {cam_fps:.1f}",
            f"face/landmark/pose = {face_fps:.1f} / {landmark_fps:.1f} / {pose_fps:.1f}",
        ]

        # 토글 상태 표시
        status_parts = []
        if self.enable_face_detection:
            status_parts.append("[1]FACE:ON")
        else:
            status_parts.append("[1]FACE:OFF")

        if self.enable_face_landmark:
            status_parts.append("[2]LAND:ON")
        else:
            status_parts.append("[2]LAND:OFF")

        if self.enable_pose_landmark:
            status_parts.append("[3]POSE:ON")
        else:
            status_parts.append("[3]POSE:OFF")

        lines.append("  ".join(status_parts))

        # 패널 배경
        pad = 6
        box_w = 520
        box_h = len(lines) * 22 + pad * 2
        bg = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160))
        screen.blit(bg, (10, 10))

        # 텍스트 렌더링
        y = 10 + pad
        for i, line in enumerate(lines):
            if i == 0:
                surf = self.font.render(line, True, (255, 255, 0))
            elif i == len(lines) - 1:
                # 토글 상태는 색상으로 표시
                surf = self.small.render(line, True, (100, 255, 100))
            else:
                surf = self.small.render(line, True, (255, 255, 255))
            screen.blit(surf, (10 + pad, y))
            y += 22

    def _draw_help_panel(self, screen, sw, sh):
        """우측 하단 도움말 패널"""
        help_lines = [
            "[H] 도움말 토글",
            "[1] Face Detection 토글",
            "[2] Face Landmark 토글",
            "[3] Pose Landmark 토글",
            f"[4] 카메라 전환 (현재: {self.camera_id})",
            "[ESC] 종료",
        ]

        pad = 8
        line_h = 20
        box_w = 220
        box_h = len(help_lines) * line_h + pad * 2

        # 우측 하단 위치
        bx = sw - box_w - 10
        by = sh - box_h - 10

        bg = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 180))
        screen.blit(bg, (bx, by))

        y = by + pad
        for line in help_lines:
            surf = self.small.render(line, True, (200, 200, 200))
            screen.blit(surf, (bx + pad, y))
            y += line_h


def main():
    parser = argparse.ArgumentParser(
        description="VisionFlow standard camera renderer"
    )
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max-fail", type=int, default=60)
    parser.add_argument("--dshow", type=int, default=1)

    # 모델 경로
    parser.add_argument(
        "--face-model-path",
        type=str,
        default="models/blaze_face_short_range.tflite",
        help="Face detector model path",
    )
    parser.add_argument(
        "--landmark-model-path",
        type=str,
        default="models/face_landmarker.task",
        help="Face landmarker model path",
    )
    parser.add_argument(
        "--pose-model-path",
        type=str,
        default="models/pose_landmarker.task",
        help="Pose landmarker model path",
    )

    args = parser.parse_args()

    app = Application(
        camera_id=args.camera_id,
        width=args.width,
        height=args.height,
        max_fail=args.max_fail,
        use_dshow=(args.dshow != 0),
        face_model_path=args.face_model_path,
        landmark_model_path=args.landmark_model_path,
        pose_model_path=args.pose_model_path,
    )
    app.run()


if __name__ == "__main__":
    main()