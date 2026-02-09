"""
file : visionFlow/utils/image.py
author : VisionFlow 개발팀

설명:
  - OpenCV 이미지와 Pygame Surface 간 변환 유틸리티 함수들
  
"""

from __future__ import annotations

import cv2
import numpy as np
import pygame


def cv_bgr_to_pygame_surface(bgr: np.ndarray) -> pygame.Surface:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = np.rot90(rgb)
    rgb = np.flipud(rgb)
    return pygame.surfarray.make_surface(rgb)
