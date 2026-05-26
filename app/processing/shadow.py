"""Shadow Removal 모듈 — HSV 기반 그림자 영역 제거

수업 외 보조 기법이지만, 오탐 감소에 필수적.
사용 이유: 그림자는 밝기(V)만 변하고 색상(H,S)은 거의 불변.
          이 특성으로 그림자 vs 실제 움직임을 구분.
"""

import cv2
import numpy as np


def remove_shadows(
    binary_mask: np.ndarray,
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    h_thresh: int = 15,
    s_thresh: int = 30,
) -> np.ndarray:
    """
    그림자 영역을 움직임 마스크에서 제거.

    원리: 두 프레임의 HSV를 비교하여
          H(색상), S(채도) 변화가 작고 V(밝기)만 변한 영역 = 그림자로 판단.

    Args:
        binary_mask: 움직임 이진 마스크 (0/255)
        prev_bgr: 이전 컬러 프레임
        curr_bgr: 현재 컬러 프레임
        h_thresh: H 채널 차이 임계값
        s_thresh: S 채널 차이 임계값

    Returns:
        cleaned_mask: 그림자 제거된 움직임 마스크
    """
    prev_hsv = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2HSV)
    curr_hsv = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2HSV)

    h_diff = cv2.absdiff(prev_hsv[:, :, 0], curr_hsv[:, :, 0])
    s_diff = cv2.absdiff(prev_hsv[:, :, 1], curr_hsv[:, :, 1])

    # H, S 변화가 작은 영역 = 그림자 (밝기만 변한 것)
    shadow_mask = (h_diff < h_thresh) & (s_diff < s_thresh)

    # 그림자 영역을 움직임 마스크에서 제거
    cleaned = binary_mask.copy()
    cleaned[shadow_mask] = 0

    return cleaned
