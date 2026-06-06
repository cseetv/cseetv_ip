"""cseetv 실험 실행기

보고서 표 4~8을 채우기 위한 실험 자동화 스크립트.
NightOwls 이미지 시퀀스 또는 MP4 영상 파일을 처리.

실행법:
  cd cseetv_ip
  python -m experiments.run_experiments --mode ablation --input data/V001/ --gt nightowls_validation.json
  python -m experiments.run_experiments --mode threshold --input data/V001/ --gt nightowls_validation.json
  python -m experiments.run_experiments --mode filters --input data/V001/ --gt nightowls_validation.json
  python -m experiments.run_experiments --mode all --input data/V001/ --gt nightowls_validation.json

입력 데이터:
  --input: PNG 이미지 폴더 또는 MP4 영상 파일 경로
  --gt: NightOwls JSON 파일 또는 VBB 파일 경로

출력:
  experiments/results/ 폴더에 CSV, JSON 저장 → 보고서 표에 직접 사용
"""

import cv2
import numpy as np
import os
import sys
import json
import csv
import time
import argparse
from glob import glob
from typing import List, Dict, Tuple, Optional

# ── 영상처리 함수들 (app.processing 모듈과 동일, standalone용 인라인) ──

def adjust_brightness(frame, target=120, clip=2.0, tile=8):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current = float(np.mean(gray))
    diff = target - current
    info = {"mu_before": round(current, 1), "correction": "normal"}

    if abs(diff) < 15:
        info["mu_after"] = round(current, 1)
        return frame.copy(), info

    adjusted = cv2.convertScaleAbs(frame, alpha=1.0, beta=diff * 0.6)

    if current < 80:
        lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        l = clahe.apply(l)
        adjusted = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        info["correction"] = "dark_corrected"
    elif current > 170:
        info["correction"] = "bright_corrected"
    else:
        info["correction"] = "soft_corrected"

    after_gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    info["mu_after"] = round(float(np.mean(after_gray)), 1)
    return adjusted, info


def denoise(frame, h=7, use_gaussian=True, gk=5, use_median=True, mk=5):
    result = frame.copy()
    if h > 0:
        result = cv2.fastNlMeansDenoisingColored(result, None, h=h, hColor=h,
                                                  templateWindowSize=7, searchWindowSize=21)
    if use_gaussian:
        k = gk if gk % 2 == 1 else gk + 1
        result = cv2.GaussianBlur(result, (k, k), 0)
    if use_median:
        k = mk if mk % 2 == 1 else mk + 1
        result = cv2.medianBlur(result, k)
    return result


def detect_motion(prev_gray, curr_gray, threshold=25, min_area=200, morph_k=5, dilate_iter=2):
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((morph_k, morph_k), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.dilate(binary, kernel, iterations=dilate_iter)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append({"x": x, "y": y, "w": w, "h": h, "area": int(area)})

    motion_px = int(np.count_nonzero(binary))
    return len(boxes) > 0, boxes, motion_px, binary


def remove_shadows(binary, prev_bgr, curr_bgr, h_thresh=15, s_thresh=30):
    prev_hsv = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2HSV)
    curr_hsv = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2HSV)
    h_diff = cv2.absdiff(prev_hsv[:, :, 0], curr_hsv[:, :, 0])
    s_diff = cv2.absdiff(prev_hsv[:, :, 1], curr_hsv[:, :, 1])
    shadow = (h_diff < h_thresh) & (s_diff < s_thresh)
    cleaned = binary.copy()
    cleaned[shadow] = 0
    return cleaned


def calc_quality(gray):
    mu = float(np.mean(gray))
    sigma = float(np.std(gray))
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    probs = hist / gray.size
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))
    return {"mu": round(mu, 1), "sigma": round(sigma, 1), "entropy": round(entropy, 2)}


# ── 프레임 로더 ──

def load_frames_from_images(folder: str, max_frames: int = 0) -> List[np.ndarray]:
    """PNG 이미지 폴더에서 프레임 로드"""
    patterns = ["*.png", "*.jpg", "*.jpeg"]
    files = []
    for p in patterns:
        files.extend(glob(os.path.join(folder, p)))
    files.sort()
    if max_frames > 0:
        files = files[:max_frames]
    print(f"[로더] {len(files)}개 이미지 로드 중...")
    frames = []
    for f in files:
        img = cv2.imread(f)
        if img is not None:
            frames.append(img)
    print(f"[로더] {len(frames)}개 프레임 로드 완료")
    return frames


def load_frames_from_video(video_path: str, fps_sample: int = 2) -> List[np.ndarray]:
    """영상 파일에서 프레임 로드 (초당 fps_sample 프레임)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상 열기 실패: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(1, int(fps / fps_sample))
    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % interval == 0:
            frames.append(frame)

    cap.release()
    print(f"[로더] 영상에서 {len(frames)}프레임 추출 (총 {count}프레임, 간격 {interval})")
    return frames


def load_frames(path: str, max_frames: int = 0) -> List[np.ndarray]:
    """경로에 따라 이미지 폴더 또는 영상 파일에서 프레임 로드"""
    if os.path.isdir(path):
        return load_frames_from_images(path, max_frames)
    elif os.path.isfile(path):
        return load_frames_from_video(path)
    else:
        raise FileNotFoundError(f"경로를 찾을 수 없음: {path}")


# ── GT 로더 ──

def load_gt(gt_path: str, num_frames: int) -> List[bool]:
    """GT 파일 로드 → bool 리스트"""
    if gt_path.endswith(".json"):
        from experiments.parse_gt import parse_nightowls_json
        gt = parse_nightowls_json(gt_path, sequence_range=(7000000, 7000000 + num_frames - 1))
        return [f["has_motion"] for f in gt["frames"]]
    elif gt_path.endswith(".vbb"):
        from experiments.parse_gt import parse_nightowls_vbb
        gt = parse_nightowls_vbb(gt_path)
        return [f["has_motion"] for f in gt["frames"][:num_frames]]
    else:
        # 수동 GT: JSON 배열 [true, false, true, ...]
        with open(gt_path) as f:
            return json.load(f)


# ── 평가 ──

def evaluate(detections: List[bool], ground_truth: List[bool]) -> Dict:
    """Precision, Recall, F1 계산"""
    n = min(len(detections), len(ground_truth))
    tp = fp = fn = tn = 0
    for i in range(n):
        d, g = detections[i], ground_truth[i]
        if d and g: tp += 1
        elif d and not g: fp += 1
        elif not d and g: fn += 1
        else: tn += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "total": n,
    }


# ── 파이프라인 실행 ──

def run_pipeline(
    frames: List[np.ndarray],
    use_clahe: bool = True,
    use_denoise: bool = True,
    use_gaussian: bool = True,
    use_median: bool = True,
    use_morph: bool = True,
    use_shadow: bool = True,
    threshold: int = 25,
    min_area: int = 200,
    target_size: Tuple[int, int] = (640, 480),
) -> Tuple[List[bool], Dict]:
    """
    프레임 리스트를 파이프라인으로 처리.

    Returns:
        (detections: [bool], stats: {mu_avg, sigma_avg, ...})
    """
    detections = []
    prev_gray = None
    prev_bgr = None
    mu_list, sigma_list = [], []
    total_time = 0

    for i, frame in enumerate(frames):
        t0 = time.time()

        # 리사이즈
        frame = cv2.resize(frame, target_size)

        # ① CLAHE
        if use_clahe:
            frame, _ = adjust_brightness(frame)

        # ② 노이즈 제거
        if use_denoise:
            frame = denoise(frame, h=7, use_gaussian=use_gaussian,
                          gk=5, use_median=use_median, mk=5)

        # Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 품질 측정
        q = calc_quality(gray)
        mu_list.append(q["mu"])
        sigma_list.append(q["sigma"])

        # ③~⑥ 움직임 감지
        if prev_gray is not None and prev_gray.shape == gray.shape:
            detected, boxes, motion_px, binary = detect_motion(
                prev_gray, gray, threshold=threshold, min_area=min_area,
                morph_k=5 if use_morph else 1,
                dilate_iter=2 if use_morph else 0,
            )

            # ⑤ Shadow Removal
            if use_shadow and prev_bgr is not None and detected:
                binary = remove_shadows(binary, prev_bgr, frame)
                # Contour 재검출
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid = [c for c in contours if cv2.contourArea(c) >= min_area]
                detected = len(valid) > 0

            detections.append(detected)
        else:
            detections.append(False)  # 첫 프레임은 비교 불가

        prev_gray = gray.copy()
        prev_bgr = frame.copy()
        total_time += time.time() - t0

    avg_time = (total_time / len(frames) * 1000) if frames else 0

    stats = {
        "mu_avg": round(np.mean(mu_list), 1) if mu_list else 0,
        "sigma_avg": round(np.mean(sigma_list), 1) if sigma_list else 0,
        "avg_time_ms": round(avg_time, 1),
        "total_frames": len(frames),
    }

    return detections, stats


# ═══════════════════════════════════════════
# 실험 1: Ablation Study (표 4)
# ═══════════════════════════════════════════

def experiment_ablation(frames, gt, threshold=25):
    """단계별 기법 적용에 따른 성능 변화"""
    print("\n" + "="*60)
    print("실험 1: 단계별 기법 적용 (Ablation Study)")
    print("="*60)

    configs = [
        {"name": "Baseline (Frame Diff만)",
         "use_clahe": False, "use_denoise": False, "use_gaussian": False,
         "use_median": False, "use_morph": False, "use_shadow": False},

        {"name": "+ CLAHE",
         "use_clahe": True, "use_denoise": False, "use_gaussian": False,
         "use_median": False, "use_morph": False, "use_shadow": False},

        {"name": "+ Gaussian + Median",
         "use_clahe": True, "use_denoise": True, "use_gaussian": True,
         "use_median": True, "use_morph": False, "use_shadow": False},

        {"name": "+ Morphology",
         "use_clahe": True, "use_denoise": True, "use_gaussian": True,
         "use_median": True, "use_morph": True, "use_shadow": False},

        {"name": "+ Shadow Removal",
         "use_clahe": True, "use_denoise": True, "use_gaussian": True,
         "use_median": True, "use_morph": True, "use_shadow": True},
    ]

    results = []
    prev_f1 = 0
    prev_fp = 0

    for i, cfg in enumerate(configs):
        name = cfg.pop("name")
        print(f"\n  [{i}] {name}...")

        detections, stats = run_pipeline(frames, threshold=threshold, **cfg)
        metrics = evaluate(detections, gt)

        delta_f1 = metrics["f1"] - prev_f1
        delta_fp = metrics["fp"] - prev_fp

        row = {
            "step": i,
            "name": name,
            "mu": stats["mu_avg"],
            "sigma": stats["sigma_avg"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "delta_f1": round(delta_f1, 4),
            "fp": metrics["fp"],
            "delta_fp": delta_fp,
            "fn": metrics["fn"],
            "time_ms": stats["avg_time_ms"],
        }
        results.append(row)

        prev_f1 = metrics["f1"]
        prev_fp = metrics["fp"]

        print(f"      P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
              f"F1={metrics['f1']:.3f} (ΔF1={delta_f1:+.3f}) "
              f"FP={metrics['fp']} FN={metrics['fn']} μ={stats['mu_avg']} σ={stats['sigma_avg']}")

    return results


# ═══════════════════════════════════════════
# 실험 2: 필터 조합 비교 (표 5)
# ═══════════════════════════════════════════

def experiment_filters(frames, gt, threshold=25):
    """노이즈 제거 필터 조합별 성능 비교"""
    print("\n" + "="*60)
    print("실험 2: 필터 조합 비교")
    print("="*60)

    configs = [
        {"name": "필터 없음", "use_denoise": False, "use_gaussian": False, "use_median": False},
        {"name": "Gaussian만", "use_denoise": True, "use_gaussian": True, "use_median": False},
        {"name": "Gaussian + Median", "use_denoise": True, "use_gaussian": True, "use_median": True},
        {"name": "Gaussian + fastNlMeans", "use_denoise": True, "use_gaussian": True, "use_median": False},
        {"name": "전부 (G+fNl+M)", "use_denoise": True, "use_gaussian": True, "use_median": True},
    ]

    results = []
    for cfg in configs:
        name = cfg.pop("name")
        print(f"\n  {name}...")

        detections, stats = run_pipeline(
            frames, use_clahe=True, use_morph=True, use_shadow=False,
            threshold=threshold, **cfg,
        )
        metrics = evaluate(detections, gt)

        results.append({
            "name": name,
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "sigma": stats["sigma_avg"],
            "time_ms": stats["avg_time_ms"],
        })

        print(f"      FP={metrics['fp']} σ={stats['sigma_avg']} "
              f"F1={metrics['f1']:.3f} time={stats['avg_time_ms']:.1f}ms")

    return results


# ═══════════════════════════════════════════
# 실험 3: 임계값 최적화 (표 6, 7)
# ═══════════════════════════════════════════

def experiment_threshold(frames, gt, thresholds=None):
    """임계값별 성능 비교"""
    if thresholds is None:
        thresholds = [10, 15, 20, 25, 30, 35, 40, 50]

    print("\n" + "="*60)
    print("실험 3: 임계값 최적화")
    print("="*60)

    results = []
    for t in thresholds:
        print(f"\n  T={t}...")

        detections, stats = run_pipeline(
            frames, use_clahe=True, use_denoise=True, use_gaussian=True,
            use_median=True, use_morph=True, use_shadow=True,
            threshold=t,
        )
        metrics = evaluate(detections, gt)

        results.append({
            "threshold": t,
            "detections": sum(detections),
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        })

        print(f"      감지={sum(detections)} FP={metrics['fp']} FN={metrics['fn']} "
              f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")

    # 최적 임계값 찾기
    best = max(results, key=lambda x: x["f1"])
    print(f"\n  ★ 최적 T={best['threshold']} (F1={best['f1']:.3f})")

    return results


# ═══════════════════════════════════════════
# 결과 저장
# ═══════════════════════════════════════════

def save_csv(data: List[Dict], path: str):
    """딕셔너리 리스트를 CSV로 저장"""
    if not data:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print(f"  → 저장: {path}")


def save_json(data, path: str):
    """JSON으로 저장"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  → 저장: {path}")


# ═══════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="cseetv 실험 실행기")
    parser.add_argument("--mode", choices=["ablation", "filters", "threshold", "all"],
                        default="all", help="실행할 실험")
    parser.add_argument("--input", required=True,
                        help="이미지 폴더 또는 영상 파일 경로")
    parser.add_argument("--gt", required=True,
                        help="GT 파일 (JSON, VBB, 또는 수동 JSON 배열)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="최대 프레임 수 (0=전체)")
    parser.add_argument("--threshold", type=int, default=25,
                        help="기본 임계값")
    parser.add_argument("--output", default="experiments/results",
                        help="결과 저장 폴더")

    args = parser.parse_args()

    # 프레임 로드
    print("\n[1/3] 프레임 로딩...")
    frames = load_frames(args.input, args.max_frames)

    if len(frames) < 2:
        print("에러: 프레임이 2개 이상 필요합니다.")
        sys.exit(1)

    # GT 로드
    print("\n[2/3] Ground Truth 로딩...")
    gt = load_gt(args.gt, len(frames))

    # GT와 프레임 수 맞추기
    min_len = min(len(frames), len(gt))
    frames = frames[:min_len]
    gt = gt[:min_len]
    print(f"  프레임: {len(frames)}, GT: {len(gt)}")

    # 실험 실행
    print(f"\n[3/3] 실험 실행 (mode={args.mode})...")

    if args.mode in ("ablation", "all"):
        results = experiment_ablation(frames, gt, args.threshold)
        save_csv(results, os.path.join(args.output, "table4_ablation.csv"))
        save_json(results, os.path.join(args.output, "table4_ablation.json"))

    if args.mode in ("filters", "all"):
        results = experiment_filters(frames, gt, args.threshold)
        save_csv(results, os.path.join(args.output, "table5_filters.csv"))

    if args.mode in ("threshold", "all"):
        results = experiment_threshold(frames, gt)
        save_csv(results, os.path.join(args.output, "table6_threshold.csv"))

    print("\n" + "="*60)
    print("모든 실험 완료!")
    print(f"결과 → {args.output}/")
    print("="*60)


if __name__ == "__main__":
    main()