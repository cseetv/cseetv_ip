"""임계값 처리 모듈 — 고정 임계값 + Adaptive Threshold

수업 연결: 기출 12번 임계값 처리
사용 이유: 야간 영상은 영역마다 밝기가 다름.
          Adaptive는 주변 픽셀 평균을 기준으로 영역별 임계값 자동 결정.
"""

import cv2
import numpy as np


def apply_threshold(
    diff: np.ndarray,
    value: int = 25,
    use_adaptive: bool = False,
    adaptive_block: int = 11,
    adaptive_c: int = 2,
) -> np.ndarray:
    """
    Frame Difference 결과를 이진화.

    Args:
        diff: absdiff 결과 (grayscale)
        value: 고정 임계값 (0~255)
        use_adaptive: True면 Adaptive Threshold 사용
        adaptive_block: Adaptive의 블록 크기 (홀수)
        adaptive_c: Adaptive의 상수 C

    Returns:
        binary: 이진화된 마스크 (0 또는 255)
    """
    if use_adaptive:
        block = adaptive_block if adaptive_block % 2 == 1 else adaptive_block + 1
        block = max(3, block)
        binary = cv2.adaptiveThreshold(
            diff, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block, -adaptive_c,
        )
    else:
        _, binary = cv2.threshold(diff, value, 255, cv2.THRESH_BINARY)

    return binary


def dynamic_threshold(base: int, brightness: float, factor: float = 0.1) -> int:
    """
    영상 밝기에 따라 임계값 자동 조정.
    어두운 장면 → 낮은 임계값, 밝은 장면 → 높은 임계값.

    Args:
        base: 기본 임계값
        brightness: 현재 프레임 평균 밝기 (0~255)
        factor: 밝기 연동 계수

    Returns:
        조정된 임계값
    """
    adjusted = base + int((brightness / 255.0) * factor * 100)
    return max(5, min(100, adjusted))
