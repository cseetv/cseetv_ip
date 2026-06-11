"""보정 파이프라인 시각화 모듈 — Contrast Stretch, Histogram EQ

수업 연결: 토픽 #8 Contrast Enhancement, #10 Histogram EQ
용도: 각 보정 단계의 결과를 이미지로 생성하여 프론트에서 나란히 표시.
"""

import cv2
import numpy as np
from ..utils.encode import gray_to_base64


def contrast_stretch(gray: np.ndarray, low_pct: float = 0.01, high_pct: float = 0.99) -> np.ndarray:
    """Contrast Stretching (1~99 percentile)"""
    flat = gray.flatten()
    flat.sort()
    lo = flat[int(len(flat) * low_pct)]
    hi = flat[int(len(flat) * high_pct)]
    rng = hi - lo if hi > lo else 1
    stretched = np.clip((gray.astype(np.float32) - lo) / rng * 255, 0, 255)
    return stretched.astype(np.uint8)


def histogram_equalization(gray: np.ndarray) -> np.ndarray:
    """전역 Histogram Equalization"""
    return cv2.equalizeHist(gray)


def get_enhancement_steps(gray: np.ndarray, quality: int = 60) -> list:
    """
    보정 파이프라인 각 단계의 결과를 base64 이미지 + 통계로 반환.
    프론트의 PipelineView 컴포넌트에서 사용.
    """
    steps = []

    def add_step(name: str, img: np.ndarray):
        mean_val = float(np.mean(img))
        std_val = float(np.std(img))
        steps.append({
            "step": name,
            "base64": gray_to_base64(img, quality),
            "mean": round(mean_val, 1),
            "std": round(std_val, 1),
        })

    add_step("원본", gray)

    cs = contrast_stretch(gray)
    add_step("Contrast Stretch", cs)

    eq = histogram_equalization(cs)
    add_step("Histogram EQ", eq)

    return steps
