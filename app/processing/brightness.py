"""밝기 보정 모듈 — CLAHE + 적응형 밝기 조절

수업 연결: 토픽 #8 Contrast Enhancement, #10 Histogram EQ
사용 이유: 야간 영상 평균 밝기 30~50. CLAHE로 지역적 대비를 높여야 객체 윤곽이 보임.
"""

import cv2
import numpy as np


def adjust_brightness(
    frame: np.ndarray,
    target_brightness: int = 120,
    clahe_clip: float = 2.0,
    clahe_tile: int = 8,
) -> tuple:
    """
    영상의 현재 밝기를 기준으로 자동 밝기 보정 수행.

    Returns:
        (adjusted_frame, info_dict)
        info_dict: { brightness_before, brightness_after, correction_type }
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current = float(np.mean(gray))

    diff = target_brightness - current

    info = {
        "brightness_before": round(current, 1),
        "brightness_after": 0.0,
        "correction_type": "normal",
    }

    # 밝기 차이가 작으면 보정하지 않음
    if abs(diff) < 15:
        info["brightness_after"] = round(current, 1)
        info["correction_type"] = "normal"
        return frame.copy(), info

    # 밝기 조절 (선형 변환)
    beta = diff * 0.6
    adjusted = cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)

    # 어두운 영상 → CLAHE로 대비 보정
    if current < 80:
        lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=(clahe_tile, clahe_tile),
        )
        enhanced_l = clahe.apply(l_ch)

        enhanced_lab = cv2.merge((enhanced_l, a_ch, b_ch))
        adjusted = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        info["correction_type"] = "dark_corrected"

    elif current > 170:
        info["correction_type"] = "bright_corrected"
    else:
        info["correction_type"] = "soft_corrected"

    after_gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    info["brightness_after"] = round(float(np.mean(after_gray)), 1)

    return adjusted, info
