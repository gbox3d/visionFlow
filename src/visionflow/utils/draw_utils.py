"""
file : visionFlow/utils/draw_utils.py
author : VisionFlow 개발팀
Pygame 기반 시각화 유틸리티

do not edit this comment
"""

from typing import List, Tuple, Dict, Any, Optional
import pygame


# ============================================================
# MediaPipe Face Mesh 연결 테이블
# ============================================================

FACE_OVAL_CONNECTIONS = [
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
]

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

# ============================================================
# MediaPipe Pose 연결 테이블
# ============================================================

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


# ============================================================
# 기본 색상 정의
# ============================================================

class Colors:
    """자주 사용하는 색상 상수"""
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    ORANGE = (255, 128, 0)
    PINK = (255, 0, 128)
    GRAY = (180, 180, 180)
    LIGHT_GRAY = (200, 200, 200)


# ============================================================
# 드로잉 유틸리티 클래스
# ============================================================

class DrawUtils:
    """
    Pygame 기반 시각화 유틸리티 클래스
    Face Detection, Face Landmark, Pose Landmark 렌더링 지원
    """

    @staticmethod
    def draw_connections(
        screen: pygame.Surface,
        landmarks: List[Dict],
        connections: List[Tuple[int, int]],
        color: Tuple[int, int, int],
        img_w: int,
        img_h: int,
        scale_x: float,
        scale_y: float,
        thickness: int = 1,
    ) -> None:
        """
        랜드마크 연결선 그리기 (Face Landmark용)
        
        Args:
            screen: pygame Surface
            landmarks: 랜드마크 리스트 [{"x": float, "y": float, "z": float}, ...]
            connections: 연결 인덱스 튜플 리스트 [(start, end), ...]
            color: RGB 색상 튜플
            img_w: 원본 이미지 폭
            img_h: 원본 이미지 높이
            scale_x: 화면 스케일 X
            scale_y: 화면 스케일 Y
            thickness: 선 두께
        """
        for start_idx, end_idx in connections:
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            x1 = int(start["x"] * img_w * scale_x)
            y1 = int(start["y"] * img_h * scale_y)
            x2 = int(end["x"] * img_w * scale_x)
            y2 = int(end["y"] * img_h * scale_y)
            pygame.draw.line(screen, color, (x1, y1), (x2, y2), thickness)

    @staticmethod
    def draw_pose_connections(
        screen: pygame.Surface,
        landmarks: List[Any],
        connections: List[Tuple[int, int]],
        color: Tuple[int, int, int],
        screen_w: int,
        screen_h: int,
        thickness: int = 2,
    ) -> None:
        """
        Pose 랜드마크 연결선 그리기
        
        Args:
            screen: pygame Surface
            landmarks: Pose 랜드마크 리스트 (객체, .x .y 속성 사용)
            connections: 연결 인덱스 튜플 리스트
            color: RGB 색상 튜플
            screen_w: 화면 폭
            screen_h: 화면 높이
            thickness: 선 두께
        """
        for a, b in connections:
            if a >= len(landmarks) or b >= len(landmarks):
                continue
            pa = landmarks[a]
            pb = landmarks[b]
            ax, ay = int(pa.x * screen_w), int(pa.y * screen_h)
            bx, by = int(pb.x * screen_w), int(pb.y * screen_h)
            pygame.draw.line(screen, color, (ax, ay), (bx, by), thickness)

    @staticmethod
    def draw_face_roi(
        screen: pygame.Surface,
        roi: Tuple[int, int, int, int],
        scale_x: float,
        scale_y: float,
        color: Tuple[int, int, int] = Colors.GREEN,
        thickness: int = 2,
        label: Optional[str] = None,
        font: Optional[pygame.font.Font] = None,
    ) -> None:
        """
        Face Detection ROI 박스 그리기
        
        Args:
            screen: pygame Surface
            roi: (x, y, width, height) 튜플
            scale_x: 화면 스케일 X
            scale_y: 화면 스케일 Y
            color: 박스 색상
            thickness: 선 두께
            label: 라벨 텍스트 (예: "Face: 0.95")
            font: pygame 폰트 (라벨 표시용)
        """
        x, y, w, h = roi
        rx = int(x * scale_x)
        ry = int(y * scale_y)
        rw = int(w * scale_x)
        rh = int(h * scale_y)

        pygame.draw.rect(screen, color, pygame.Rect(rx, ry, rw, rh), thickness)

        if label and font:
            label_surf = font.render(label, True, color)
            screen.blit(label_surf, (rx, ry - 18))

    @staticmethod
    def draw_face_landmarks(
        screen: pygame.Surface,
        face_landmarks_data: List[List[Dict]],
        img_w: int,
        img_h: int,
        scale_x: float,
        scale_y: float,
        draw_points: bool = True,
        draw_oval: bool = True,
        draw_eyes: bool = True,
        draw_lips: bool = True,
        point_color: Tuple[int, int, int] = Colors.WHITE,
        oval_color: Tuple[int, int, int] = Colors.GRAY,
        eye_color: Tuple[int, int, int] = Colors.CYAN,
        lips_color: Tuple[int, int, int] = Colors.PINK,
        point_radius: int = 1,
        line_thickness: int = 1,
    ) -> None:
        """
        Face Landmark 전체 그리기
        
        Args:
            screen: pygame Surface
            face_landmarks_data: 얼굴 랜드마크 데이터 리스트
            img_w: 원본 이미지 폭
            img_h: 원본 이미지 높이
            scale_x: 화면 스케일 X
            scale_y: 화면 스케일 Y
            draw_points: 점 그리기 여부
            draw_oval: 얼굴 윤곽 그리기 여부
            draw_eyes: 눈 그리기 여부
            draw_lips: 입술 그리기 여부
            point_color: 점 색상
            oval_color: 윤곽 색상
            eye_color: 눈 색상
            lips_color: 입술 색상
            point_radius: 점 반지름
            line_thickness: 선 두께
        """
        if not face_landmarks_data:
            return

        for face_lms in face_landmarks_data:
            # 랜드마크 점 그리기
            if draw_points:
                for lm in face_lms:
                    px = int(lm["x"] * img_w * scale_x)
                    py = int(lm["y"] * img_h * scale_y)
                    pygame.draw.circle(screen, point_color, (px, py), point_radius)

            # 연결선 그리기
            if draw_oval:
                DrawUtils.draw_connections(
                    screen, face_lms, FACE_OVAL_CONNECTIONS,
                    oval_color, img_w, img_h, scale_x, scale_y, line_thickness
                )
            if draw_eyes:
                DrawUtils.draw_connections(
                    screen, face_lms, LEFT_EYE_CONNECTIONS,
                    eye_color, img_w, img_h, scale_x, scale_y, line_thickness
                )
                DrawUtils.draw_connections(
                    screen, face_lms, RIGHT_EYE_CONNECTIONS,
                    eye_color, img_w, img_h, scale_x, scale_y, line_thickness
                )
            if draw_lips:
                DrawUtils.draw_connections(
                    screen, face_lms, LIPS_CONNECTIONS,
                    lips_color, img_w, img_h, scale_x, scale_y, line_thickness
                )

    @staticmethod
    def draw_pose_landmarks(
        screen: pygame.Surface,
        pose_landmarks_data: List[List[Any]],
        screen_w: int,
        screen_h: int,
        draw_points: bool = True,
        draw_connections: bool = True,
        line_color: Tuple[int, int, int] = Colors.ORANGE,
        point_color: Tuple[int, int, int] = Colors.RED,
        point_radius: int = 4,
        line_thickness: int = 2,
    ) -> None:
        """
        Pose Landmark 전체 그리기
        
        Args:
            screen: pygame Surface
            pose_landmarks_data: 포즈 랜드마크 데이터 리스트
            screen_w: 화면 폭
            screen_h: 화면 높이
            draw_points: 점 그리기 여부
            draw_connections: 연결선 그리기 여부
            line_color: 연결선 색상
            point_color: 점 색상
            point_radius: 점 반지름
            line_thickness: 선 두께
        """
        if not pose_landmarks_data:
            return

        for landmarks in pose_landmarks_data:
            # 연결선 그리기
            if draw_connections:
                DrawUtils.draw_pose_connections(
                    screen, landmarks, POSE_CONNECTIONS,
                    line_color, screen_w, screen_h, line_thickness
                )

            # 점 그리기
            if draw_points:
                for lm in landmarks:
                    x = int(lm.x * screen_w)
                    y = int(lm.y * screen_h)
                    pygame.draw.circle(screen, point_color, (x, y), point_radius)

    @staticmethod
    def draw_info_box(
        screen: pygame.Surface,
        lines: List[Tuple[str, Tuple[int, int, int], pygame.font.Font]],
        x: int,
        y: int,
        padding: int = 6,
        bg_alpha: int = 160,
        min_width: int = 0,
    ) -> None:
        """
        정보 박스 그리기 (반투명 배경 + 텍스트)
        
        Args:
            screen: pygame Surface
            lines: [(텍스트, 색상, 폰트), ...] 리스트
            x: 박스 X 좌표
            y: 박스 Y 좌표
            padding: 내부 여백
            bg_alpha: 배경 투명도 (0~255)
            min_width: 최소 박스 폭
        """
        if not lines:
            return

        # 박스 크기 계산
        line_height = 22
        max_width = min_width
        for text, _, font in lines:
            text_width = font.size(text)[0]
            max_width = max(max_width, text_width)

        box_w = max_width + padding * 2
        box_h = len(lines) * line_height + padding * 2

        # 반투명 배경
        bg = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, bg_alpha))
        screen.blit(bg, (x, y))

        # 텍스트 렌더링
        curr_y = y + padding
        for text, color, font in lines:
            surf = font.render(text, True, color)
            screen.blit(surf, (x + padding, curr_y))
            curr_y += line_height
