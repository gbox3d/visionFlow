"""
file : visionFlow/utils/face_geometry.py
author : VisionFlow 개발팀
얼굴 3D 기하학 연산 유틸리티

do not edit this comment
"""

import math
import numpy as np

class FaceGeometryUtils:
    """
    얼굴 3D 기하학 연산(거리, 각도)을 담당하는 유틸리티 클래스
    """

    @staticmethod
    def matrix_to_euler_angles(R: np.ndarray) -> tuple[float, float, float]:
        """
        4x4 또는 3x3 회전 행렬을 Euler Angles (Pitch, Yaw, Roll)로 변환
        
        Args:
            R: 변환 행렬 (numpy array)
        Returns:
            (pitch, yaw, roll) 단위: Degree
        """
        # 4x4 행렬인 경우 3x3 회전 부분만 추출
        if R.shape == (4, 4):
            R = R[:3, :3]
            
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.degrees(x), np.degrees(y), np.degrees(z)

    @staticmethod
    def estimate_distance_from_eyes(
        landmarks: list[dict],
        image_width: int,
        image_height: int,
        yaw_deg: float,
        focal_length_constant: float = 650.0,
        real_eye_dist_cm: float = 6.3
    ) -> float:
        """
        눈 사이 거리(IPD)와 Yaw 각도를 이용해 카메라와의 거리를 추정
        
        Args:
            landmarks: 랜드마크 리스트 (normalized x, y)
            image_width: 이미지 폭
            image_height: 이미지 높이
            yaw_deg: 현재 얼굴의 Yaw 각도 (회전 보정용)
            focal_length_constant: 카메라 초점거리 상수 (캘리브레이션 필요)
            real_eye_dist_cm: 실제 사람 눈 간 평균 거리 (cm)
            
        Returns:
            distance (cm)
        """
        if not landmarks or len(landmarks) <= 263:
            return 0.0

        # 눈꼬리 랜드마크 인덱스 (MediaPipe FaceMesh 기준)
        # 33: 왼쪽 눈꼬리, 263: 오른쪽 눈꼬리
        idx_l, idx_r = 33, 263
        
        p1 = landmarks[idx_l]
        p2 = landmarks[idx_r]

        # 정규 좌표 -> 픽셀 좌표 변환
        x1, y1 = p1["x"] * image_width, p1["y"] * image_height
        x2, y2 = p2["x"] * image_width, p2["y"] * image_height

        # 화면상 픽셀 거리 (유클리드 거리)
        pixel_dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if pixel_dist == 0:
            return 0.0

        # [Yaw 보정]
        # 고개를 돌리면 눈 사이가 좁아 보이므로, cos(yaw)로 나누어 정면 기준 너비로 복원
        yaw_rad = math.radians(yaw_deg)
        cos_val = abs(math.cos(yaw_rad))
        
        # 90도에 가까워지면 값이 폭주하므로 최소값 클램핑 (약 78도까지만 신뢰)
        if cos_val < 0.2: 
            cos_val = 0.2

        corrected_pixel_dist = pixel_dist / cos_val

        # 거리 공식: Distance = (Focal_Length * Real_Width) / Pixel_Width
        distance_cm = (focal_length_constant * real_eye_dist_cm) / corrected_pixel_dist
        
        return distance_cm