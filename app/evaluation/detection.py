"""감지 성능 + 노이즈 + ROC 평가 지표

감지 성능: Precision, Recall, F1-Score
노이즈: SNR, PSNR
ROC: 임계값별 TPR/FPR
"""

import numpy as np
from typing import List, Dict


# ── 감지 성능 ──

def calc_detection_metrics(
    detections: List[bool],
    ground_truth: List[bool],
) -> Dict:
    """
    프레임 단위 감지 결과 vs 정답 비교.

    Args:
        detections: 프레임별 감지 여부 [True, False, True, ...]
        ground_truth: 프레임별 실제 움직임 여부

    Returns:
        { tp, fp, fn, tn, precision, recall, f1 }
    """
    tp = fp = fn = tn = 0

    for det, gt in zip(detections, ground_truth):
        if det and gt:
            tp += 1
        elif det and not gt:
            fp += 1
        elif not det and gt:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ── 노이즈 지표 ──

def calc_snr(original: np.ndarray, processed: np.ndarray) -> float:
    """
    SNR (Signal-to-Noise Ratio) 계산.
    SNR = 10 × log₁₀(σ²_signal / σ²_noise)
    noise = original - processed
    """
    orig = original.astype(np.float64)
    proc = processed.astype(np.float64)
    noise = orig - proc
    signal_power = np.var(proc)
    noise_power = np.var(noise)
    if noise_power < 1e-10:
        return 100.0
    return round(float(10 * np.log10(signal_power / noise_power)), 2)


def calc_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """
    PSNR (Peak Signal-to-Noise Ratio) 계산.
    PSNR = 10 × log₁₀(MAX² / MSE)
    30dB 이상 양호, 40dB 이상 우수.
    """
    mse = float(np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2))
    if mse < 1e-10:
        return 100.0
    return round(float(10 * np.log10(255.0 ** 2 / mse)), 2)


# ── ROC 곡선 ──

def generate_roc_data(
    diff_values: List[float],
    ground_truth: List[bool],
    thresholds: List[int] = None,
) -> List[Dict]:
    """
    임계값별 TPR/FPR 계산하여 ROC 곡선 데이터 생성.

    Args:
        diff_values: 각 프레임의 motion_ratio (0~1)
        ground_truth: 각 프레임의 실제 움직임 여부
        thresholds: 테스트할 임계값 리스트

    Returns:
        [{ "threshold": T, "tpr": float, "fpr": float }, ...]
    """
    if thresholds is None:
        thresholds = list(range(5, 85, 5))

    results = []
    for t in thresholds:
        scaled_t = t / 2500.0  # risk_score 역변환
        detections = [v > scaled_t for v in diff_values]

        tp = fp = fn = tn = 0
        for det, gt in zip(detections, ground_truth):
            if det and gt:
                tp += 1
            elif det and not gt:
                fp += 1
            elif not det and gt:
                fn += 1
            else:
                tn += 1

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        results.append({
            "threshold": t,
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
        })

    return results


def calc_auc(roc_data: List[Dict]) -> float:
    """ROC 곡선의 AUC (Area Under Curve) 계산"""
    sorted_data = sorted(roc_data, key=lambda x: x["fpr"])
    auc = 0.0
    for i in range(1, len(sorted_data)):
        dx = sorted_data[i]["fpr"] - sorted_data[i - 1]["fpr"]
        avg_y = (sorted_data[i]["tpr"] + sorted_data[i - 1]["tpr"]) / 2
        auc += dx * avg_y
    return round(auc, 4)
