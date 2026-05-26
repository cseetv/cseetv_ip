"""움직임 감지 모듈 — Frame Difference + Morphology + Contour

수업 연결: 토픽 #13 Image Difference, 기출 12번 모폴로지
사용 이유:
  - absdiff: 연속 프레임 차이로 움직임 검출. 가장 기본적이고 연산 가벼움.
  - MORPH_OPEN: 작은 노이즈 덩어리 제거 (침식→팽창).
  - Dilate: 움직임 영역 확장 → contour 검출 안정화.
  - Contour Area: 일정 크기 이상만 유효 움직임으로 판단.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
from .threshold import apply_threshold, dynamic_threshold


def detect_motion(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    threshold_value: int = 25,
    use_adaptive: bool = False,
    use_dynamic: bool = False,
    dynamic_base: int = 25,
    dynamic_factor: float = 0.1,
    brightness: float = 120.0,
    min_area: int = 200,
    morph_kernel_size: int = 5,
    dilate_iterations: int = 2,
) -> Dict:
    """
    두 프레임 간 움직임 감지.

    Returns:
        {
            "detected": bool,
            "boxes": [{"x","y","w","h","area"}],
            "total_motion_pixels": int,
            "diff_gray": ndarray (시각화용),
            "binary_mask": ndarray (후속 처리용),
        }
    """
    # Frame Difference
    diff = cv2.absdiff(prev_gray, curr_gray)

    # 동적 임계값 적용
    if use_dynamic:
        threshold_value = dynamic_threshold(dynamic_base, brightness, dynamic_factor)

    # 이진화
    binary = apply_threshold(diff, threshold_value, use_adaptive)

    # Morphology
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)

    # MORPH_OPEN: 작은 노이즈 제거
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Dilate: 움직임 영역 확장
    binary = cv2.dilate(binary, kernel, iterations=dilate_iterations)

    # Contour 검출 + Area 필터링
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area)})

    motion_pixels = int(np.count_nonzero(binary))

    return {
        "detected": len(boxes) > 0,
        "boxes": boxes,
        "total_motion_pixels": motion_pixels,
        "diff_gray": diff,
        "binary_mask": binary,
        "threshold_used": threshold_value,
    }


class TemporalSmoother:
    """
    시간축 안정화: 연속 N프레임에서 같은 위치에 움직임이 있어야 진짜로 판단.
    순간적 조명 변화(번개, 헤드라이트)에 의한 단발성 오탐 제거.
    """

    def __init__(self, required_frames: int = 3):
        self.required = required_frames
        self.history: List[bool] = []

    def update(self, detected: bool) -> bool:
        """검출 결과 추가 후, 연속 N프레임 감지 여부 반환"""
        self.history.append(detected)
        if len(self.history) > self.required:
            self.history.pop(0)

        # 최근 N프레임 모두 감지되었는지
        if len(self.history) < self.required:
            return detected

        return all(self.history[-self.required:])

    def reset(self):
        self.history.clear()
