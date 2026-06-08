"""cseetv 통합 실험 실행기

CDnet 데이터셋 (tramStation 등) + 직접 촬영 영상 지원.
app/processing/ 모듈을 import → 웹 서버와 동일한 파이프라인.

═══ CDnet (tramStation) ═══

  # 빠른 테스트 (200프레임)
  python -m experiments.run --source cdnet --data data/tramStation --max-frames 200

  # 전체 (3,000프레임)
  python -m experiments.run --source cdnet --data data/tramStation

  # 특정 실험만
  python -m experiments.run --source cdnet --data data/tramStation --mode ablation

═══ 직접 촬영 영상 ═══

  python -m experiments.run --source video --video data/D4.mp4 --gt data/D4_gt.json
"""

import cv2
import numpy as np
import os, sys, csv, time, argparse
from typing import List, Dict, Tuple, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.processing.brightness import adjust_brightness
from app.processing.denoising import denoise
from app.processing.motion import detect_motion
from app.processing.shadow import remove_shadows
from app.evaluation.quality import calc_quality_metrics
from app.evaluation.detection import calc_detection_metrics
from experiments.loaders import load_cdnet, load_video, evaluate_cdnet_pixel


# ═══ 파이프라인 (app/processing 사용) ═══

def run_pipeline(
    frames, gt_masks=None,
    use_clahe=True, use_denoise=True,
    use_gaussian=True, use_median=True,
    use_morph=True, use_shadow=True,
    threshold=25, min_area=200,
) -> Dict:
    """
    파이프라인 실행 → 프레임 단위 + 픽셀 단위 평가 결과 반환.

    gt_masks: CDnet GT 마스크 리스트 (있으면 픽셀 단위 평가)
    """
    detections = []  # 프레임별 bool
    prev_gray, prev_bgr = None, None
    mu_list, sigma_list = [], []
    pixel_tp, pixel_fp, pixel_fn, pixel_tn = 0, 0, 0, 0
    t0 = time.time()

    for i, frame in enumerate(frames):
        # 리사이즈 하지 않음 — CDnet 원본 크기(480x295) 유지
        # 단, GT 마스크와 크기가 일치해야 함

        # ① brightness.py
        if use_clahe:
            frame, _ = adjust_brightness(frame, target_brightness=120, clahe_clip=2.0, clahe_tile=8)

        # ② denoising.py
        if use_denoise:
            frame = denoise(frame, denoise_h=7,
                          use_gaussian=use_gaussian, gaussian_kernel=5,
                          use_median=use_median, median_kernel=5)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        q = calc_quality_metrics(gray)
        mu_list.append(q["mean"])
        sigma_list.append(q["std"])

        # ③④ motion.py
        detected = False
        binary_mask = np.zeros_like(gray)

        if prev_gray is not None and prev_gray.shape == gray.shape:
            result = detect_motion(
                prev_gray, gray,
                threshold_value=threshold, min_area=min_area,
                morph_kernel_size=5 if use_morph else 1,
                dilate_iterations=2 if use_morph else 0,
            )
            detected = result["detected"]
            binary_mask = result["binary_mask"]

            # ⑤ shadow.py
            if use_shadow and prev_bgr is not None:
                binary_mask = remove_shadows(binary_mask, prev_bgr, frame)
                contours, _ = cv2.findContours(
                    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                detected = any(cv2.contourArea(c) >= min_area for c in contours)

        detections.append(detected)

        # 픽셀 단위 평가 (CDnet GT 마스크가 있을 때)
        if gt_masks and i < len(gt_masks) and gt_masks[i] is not None:
            gt_m = gt_masks[i]
            # 크기 맞추기
            if binary_mask.shape != gt_m.shape:
                binary_mask = cv2.resize(binary_mask, (gt_m.shape[1], gt_m.shape[0]))
            px = evaluate_cdnet_pixel(binary_mask, gt_m)
            pixel_tp += px["tp"]
            pixel_fp += px["fp"]
            pixel_fn += px["fn"]
            pixel_tn += px["tn"]

        prev_gray = gray.copy()
        prev_bgr = frame.copy()

        if (i + 1) % 300 == 0:
            print(f"    {i+1}/{len(frames)} ({(i+1)/len(frames)*100:.0f}%)")

    elapsed = time.time() - t0

    # 픽셀 단위 지표
    px_prec = pixel_tp / max(pixel_tp + pixel_fp, 1)
    px_rec = pixel_tp / max(pixel_tp + pixel_fn, 1)
    px_f1 = 2 * px_prec * px_rec / max(px_prec + px_rec, 1e-10)

    return {
        "detections": detections,
        "mu": round(np.mean(mu_list), 1) if mu_list else 0,
        "sigma": round(np.mean(sigma_list), 1) if sigma_list else 0,
        "time_ms": round(elapsed / max(len(frames), 1) * 1000, 1),
        # 픽셀 단위
        "px_precision": round(px_prec, 4),
        "px_recall": round(px_rec, 4),
        "px_f1": round(px_f1, 4),
        "px_tp": pixel_tp, "px_fp": pixel_fp,
        "px_fn": pixel_fn, "px_tn": pixel_tn,
    }


# ═══ 실험 1: Ablation Study ═══

def exp_ablation(frames, gt_masks, gt_binary, T=25):
    print("\n" + "=" * 60)
    print("실험 1: Ablation Study (표 4)")
    print("=" * 60)

    configs = [
        ("Baseline (Frame Diff만)",
         dict(use_clahe=False, use_denoise=False, use_gaussian=False,
              use_median=False, use_morph=False, use_shadow=False)),
        ("+ CLAHE",
         dict(use_clahe=True, use_denoise=False, use_gaussian=False,
              use_median=False, use_morph=False, use_shadow=False)),
        ("+ Gaussian + Median",
         dict(use_clahe=True, use_denoise=True, use_gaussian=True,
              use_median=True, use_morph=False, use_shadow=False)),
        ("+ Morphology",
         dict(use_clahe=True, use_denoise=True, use_gaussian=True,
              use_median=True, use_morph=True, use_shadow=False)),
        ("+ Shadow Removal (최종)",
         dict(use_clahe=True, use_denoise=True, use_gaussian=True,
              use_median=True, use_morph=True, use_shadow=True)),
    ]

    rows = []
    prev_f1 = 0

    for i, (name, cfg) in enumerate(configs):
        print(f"\n  [{i}] {name}...")
        r = run_pipeline(frames, gt_masks, threshold=T, **cfg)

        # 프레임 단위 평가
        fm = calc_detection_metrics(r["detections"], gt_binary)

        df1 = round(r["px_f1"] - prev_f1, 4)
        rows.append({
            "step": i, "name": name,
            "mu": r["mu"], "sigma": r["sigma"],
            # 프레임 단위
            "frame_prec": fm["precision"], "frame_rec": fm["recall"], "frame_f1": fm["f1"],
            # 픽셀 단위 (더 정확)
            "px_prec": r["px_precision"], "px_rec": r["px_recall"], "px_f1": r["px_f1"],
            "delta_f1": f"{df1:+.4f}",
            "px_fp": r["px_fp"], "px_fn": r["px_fn"],
            "time_ms": r["time_ms"],
        })
        prev_f1 = r["px_f1"]

        print(f"      [픽셀] P={r['px_precision']:.3f} R={r['px_recall']:.3f} F1={r['px_f1']:.3f} (ΔF1={df1:+.3f})")
        print(f"      [프레임] P={fm['precision']:.3f} R={fm['recall']:.3f} F1={fm['f1']:.3f}")

    return rows


# ═══ 실험 2: 필터 비교 ═══

def exp_filters(frames, gt_masks, gt_binary, T=25):
    print("\n" + "=" * 60)
    print("실험 2: 필터 조합 비교 (표 5)")
    print("=" * 60)

    configs = [
        ("필터 없음", dict(use_denoise=False, use_gaussian=False, use_median=False)),
        ("Gaussian만", dict(use_denoise=True, use_gaussian=True, use_median=False)),
        ("Gaussian+Median", dict(use_denoise=True, use_gaussian=True, use_median=True)),
        ("전부(G+fNl+M)", dict(use_denoise=True, use_gaussian=True, use_median=True)),
    ]

    rows = []
    for name, cfg in configs:
        print(f"\n  {name}...")
        r = run_pipeline(frames, gt_masks, use_clahe=True, use_morph=True,
                         use_shadow=False, threshold=T, **cfg)
        rows.append({
            "name": name,
            "px_prec": r["px_precision"], "px_rec": r["px_recall"], "px_f1": r["px_f1"],
            "sigma": r["sigma"], "time_ms": r["time_ms"],
        })
        print(f"      P={r['px_precision']:.3f} R={r['px_recall']:.3f} "
              f"F1={r['px_f1']:.3f} time={r['time_ms']:.1f}ms")

    return rows


# ═══ 실험 3: 임계값 최적화 ═══

def exp_threshold(frames, gt_masks, gt_binary):
    print("\n" + "=" * 60)
    print("실험 3: 임계값 최적화 (표 6)")
    print("=" * 60)

    Ts = [10, 15, 20, 25, 30, 35, 40, 50]
    rows = []

    for t in Ts:
        print(f"\n  T={t}...")
        r = run_pipeline(frames, gt_masks, threshold=t)
        fm = calc_detection_metrics(r["detections"], gt_binary)
        rows.append({
            "threshold": t,
            "px_prec": r["px_precision"], "px_rec": r["px_recall"], "px_f1": r["px_f1"],
            "frame_prec": fm["precision"], "frame_rec": fm["recall"], "frame_f1": fm["f1"],
        })
        print(f"      [픽셀] P={r['px_precision']:.3f} R={r['px_recall']:.3f} F1={r['px_f1']:.3f}")

    best = max(rows, key=lambda x: x["px_f1"])
    print(f"\n  ★ 최적 T={best['threshold']} (픽셀 F1={best['px_f1']:.3f})")
    return rows


# ═══ 저장 ═══

def save_csv(data, path):
    if not data: return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=data[0].keys())
        w.writeheader()
        w.writerows(data)
    print(f"  → {path}")


# ═══ 메인 ═══

def main():
    p = argparse.ArgumentParser(description="cseetv 통합 실험 실행기")
    p.add_argument("--source", choices=["cdnet", "video"], required=True)
    p.add_argument("--mode", choices=["ablation", "filters", "threshold", "all"], default="all")

    # CDnet 옵션
    p.add_argument("--data", default="data/tramStation", help="CDnet 데이터셋 폴더")

    # 영상 옵션
    p.add_argument("--video", default=None)
    p.add_argument("--gt", default=None)

    # 공통
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--threshold", type=int, default=25)
    p.add_argument("--output", default="experiments/results")

    args = p.parse_args()

    # ── 데이터 로드 ──
    gt_masks = None

    if args.source == "cdnet":
        data = load_cdnet(args.data, args.max_frames)
        frames = data["frames"]
        gt_masks = data["gt_masks"]
        gt_binary = data["gt_binary"]
        tag = os.path.basename(args.data)

    elif args.source == "video":
        if not args.video:
            print("--video 필요"); sys.exit(1)
        frames, gt_binary = load_video(args.video, args.gt, max_frames=args.max_frames)
        tag = os.path.splitext(os.path.basename(args.video))[0]

    if len(frames) < 2:
        print("프레임 부족"); sys.exit(1)

    has_pixel_gt = gt_masks is not None and any(m is not None for m in gt_masks)
    print(f"\n  프레임: {len(frames)}개")
    print(f"  GT: {'픽셀 단위 (CDnet)' if has_pixel_gt else '프레임 단위'}")
    print(f"  파이프라인: app.processing (웹 서버와 동일)\n")

    # ── 실험 ──
    if args.mode in ("ablation", "all"):
        r = exp_ablation(frames, gt_masks, gt_binary, args.threshold)
        save_csv(r, os.path.join(args.output, f"table4_ablation_{tag}.csv"))

    if args.mode in ("filters", "all"):
        r = exp_filters(frames, gt_masks, gt_binary, args.threshold)
        save_csv(r, os.path.join(args.output, f"table5_filters_{tag}.csv"))

    if args.mode in ("threshold", "all"):
        r = exp_threshold(frames, gt_masks, gt_binary)
        save_csv(r, os.path.join(args.output, f"table6_threshold_{tag}.csv"))

    print("\n" + "=" * 60)
    print(f"완료! → {args.output}/")
    print("=" * 60)


if __name__ == "__main__":
    main()