"""영상 품질 진단 지표 — Mean, Std, Entropy, Histogram

매 프레임마다 실시간 계산. 보정 파이프라인이 잘 동작하는지 확인용.
"""

import cv2
import numpy as np


def calc_quality_metrics(gray: np.ndarray) -> dict:
    """
    그레이스케일 영상의 품질 지표 계산.

    Returns:
        {
            "mean": 평균 밝기 (0~255),
            "std": 밝기 표준편차 (대비 정도),
            "entropy": 엔트로피 (정보량, bit),
            "histogram": 256개 빈도 리스트,
            "diagnosis": "under_exposed" | "over_exposed" | "low_contrast" | "good"
        }
    """
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_list = hist.tolist()

    total = gray.size
    probs = hist / total
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))

    left_ratio = float(np.sum(hist[:85])) / total
    right_ratio = float(np.sum(hist[170:])) / total

    if left_ratio > 0.6:
        diagnosis = "under_exposed"
    elif right_ratio > 0.6:
        diagnosis = "over_exposed"
    elif std_val < 30:
        diagnosis = "low_contrast"
    else:
        diagnosis = "good"

    return {
        "mean": round(mean_val, 1),
        "std": round(std_val, 1),
        "entropy": round(entropy, 2),
        "histogram": hist_list,
        "diagnosis": diagnosis,
    }
