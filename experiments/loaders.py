"""데이터 로더 — CDnet / 영상 파일 / 수동 GT 통합

CDnet 형식:
  input/in000001.jpg~in003000.jpg  (원본 프레임)
  groundtruth/gt000001.png~gt003000.png  (GT 마스크)
  temporalROI.txt  (평가 구간: "start end")
  ROI.bmp  (공간 ROI 마스크)

GT 마스크 픽셀값:
  0   = Static (배경)
  50  = Hard shadow (평가 제외 또는 별도)
  85  = Outside ROI (평가 제외)
  170 = Unknown (평가 제외)
  255 = Motion (전경/움직임)
"""

import cv2
import json
import os
import numpy as np
from glob import glob
from typing import List, Tuple, Optional, Dict


# ═══ CDnet 로더 ═══

def load_cdnet(
    dataset_dir: str,
    max_frames: int = 0,
) -> Dict:
    """
    CDnet 데이터셋 로드.

    Args:
        dataset_dir: tramStation/ 폴더 경로
        max_frames: 최대 프레임 (0=전체)

    Returns:
        {
            "frames": [ndarray],        # 입력 프레임 (BGR)
            "gt_masks": [ndarray],      # GT 마스크 (grayscale)
            "gt_binary": [bool],        # 프레임별 움직임 여부
            "roi_mask": ndarray | None, # 공간 ROI
            "temporal_roi": (int, int), # 평가 구간 (start, end)
            "total_frames": int,
            "eval_start": int,
            "eval_end": int,
        }
    """
    input_dir = os.path.join(dataset_dir, "input")
    gt_dir = os.path.join(dataset_dir, "groundtruth")

    # 프레임 파일 목록 (번호순 정렬)
    input_files = sorted(glob(os.path.join(input_dir, "in*.jpg")) +
                         glob(os.path.join(input_dir, "in*.png")))
    gt_files = sorted(glob(os.path.join(gt_dir, "gt*.png")))

    print(f"[CDnet] 입력: {len(input_files)}프레임, GT: {len(gt_files)}프레임")

    # temporalROI.txt 읽기 (평가 구간)
    temporal_path = os.path.join(dataset_dir, "temporalROI.txt")
    eval_start, eval_end = 1, len(input_files)
    if os.path.exists(temporal_path):
        with open(temporal_path, encoding="utf-8", errors="ignore") as f:
            parts = f.read().strip().split()
            if len(parts) >= 2:
                eval_start, eval_end = int(parts[0]), int(parts[1])
        print(f"[CDnet] 평가 구간: 프레임 {eval_start}~{eval_end}")

    # 공간 ROI 마스크
    roi_mask = None
    for roi_name in ["ROI.bmp", "ROI.jpg", "ROI.png"]:
        roi_path = os.path.join(dataset_dir, roi_name)
        if os.path.exists(roi_path):
            roi_mask = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
            print(f"[CDnet] ROI 마스크 로드: {roi_name} ({roi_mask.shape})")
            break

    # 프레임 + GT 로드 (평가 구간만)
    start_idx = max(0, eval_start - 1)
    end_idx = min(len(input_files), eval_end)

    if max_frames > 0:
        end_idx = min(end_idx, start_idx + max_frames)

    frames = []
    gt_masks = []
    gt_binary = []

    for i in range(start_idx, end_idx):
        # 입력 프레임
        if i >= len(input_files):
            break
        frame = cv2.imread(input_files[i])
        if frame is None:
            continue

        # GT 마스크
        gt_mask = None
        if i < len(gt_files):
            gt_mask = cv2.imread(gt_files[i], cv2.IMREAD_GRAYSCALE)

        frames.append(frame)

        if gt_mask is not None:
            gt_masks.append(gt_mask)
            # 픽셀값 255 = Motion → 프레임에 움직임 있음
            has_motion = np.any(gt_mask == 255)
            gt_binary.append(bool(has_motion))
        else:
            gt_masks.append(None)
            gt_binary.append(False)

        if len(frames) % 500 == 0:
            print(f"  ... {len(frames)}프레임 로드됨")

    positive = sum(gt_binary)
    print(f"[CDnet] 로드 완료: {len(frames)}프레임")
    print(f"  양성(움직임): {positive} ({positive/max(len(gt_binary),1)*100:.1f}%)")
    print(f"  음성(정지):   {len(gt_binary)-positive}")

    return {
        "frames": frames,
        "gt_masks": gt_masks,
        "gt_binary": gt_binary,
        "roi_mask": roi_mask,
        "temporal_roi": (eval_start, eval_end),
        "total_frames": len(frames),
        "eval_start": eval_start,
        "eval_end": eval_end,
    }


def evaluate_cdnet_pixel(
    detection_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> Dict:
    """
    CDnet 픽셀 단위 평가.

    GT 픽셀값:
      0   = Static → 감지하면 FP
      255 = Motion → 감지하면 TP, 못하면 FN
      50, 85, 170 = 평가 제외

    Returns: { tp, fp, fn, tn }
    """
    # 평가 대상 픽셀만 (0 또는 255)
    eval_mask = (gt_mask == 0) | (gt_mask == 255)

    gt_positive = (gt_mask == 255) & eval_mask
    gt_negative = (gt_mask == 0) & eval_mask

    det = detection_mask > 0  # 감지된 픽셀

    tp = int(np.sum(det & gt_positive))
    fp = int(np.sum(det & gt_negative))
    fn = int(np.sum(~det & gt_positive))
    tn = int(np.sum(~det & gt_negative))

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ═══ 영상 파일 로더 ═══

def load_video(
    video_path: str,
    gt_path: Optional[str] = None,
    fps_sample: int = 2,
    max_frames: int = 0,
) -> Tuple[List[np.ndarray], List[bool]]:
    """직접 촬영 영상 로드."""
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
            if max_frames > 0 and len(frames) >= max_frames:
                break
    cap.release()
    print(f"[영상] {len(frames)}프레임 추출 (원본 {count}프레임)")

    gt = [False] * len(frames)
    if gt_path and os.path.exists(gt_path):
        gt = _load_gt_json(gt_path, len(frames), fps, interval)
        print(f"[GT] 양성 {sum(gt)}, 음성 {len(gt)-sum(gt)}")
    else:
        print("[GT] 없음 — 품질 지표만 측정")

    return frames, gt


def _load_gt_json(path, num_frames, fps, interval):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        gt = data[:num_frames]
        while len(gt) < num_frames:
            gt.append(False)
        return gt
    if "segments" in data:
        gt = []
        for seg in data["segments"]:
            n = int((seg["end_sec"] - seg["start_sec"]) * fps / interval)
            gt.extend([seg["motion"]] * n)
        return gt[:num_frames]
    raise ValueError(f"지원하지 않는 GT: {path}")