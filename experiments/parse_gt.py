"""NightOwls Ground Truth 파서

COCO JSON 형식과 VBB(MATLAB) 형식 모두 지원.
프레임별 "움직임 있음/없음"으로 변환하여 평가에 사용.

사용법:
  gt = parse_nightowls_json("nightowls_validation.json", sequence_range=(7000000, 7001317))
  gt = parse_nightowls_vbb("V001.vbb")
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional


def parse_nightowls_json(
    json_path: str,
    sequence_range: Optional[Tuple[int, int]] = None,
) -> Dict:
    """
    NightOwls COCO JSON → 프레임별 ground truth 변환.

    Args:
        json_path: nightowls_validation.json 경로
        sequence_range: (start_image_id, end_image_id) — 특정 시퀀스만 추출
                        None이면 전체 사용

    Returns:
        {
            "frames": [
                {
                    "image_id": 7000000,
                    "file_name": "xxx.png",
                    "has_motion": True,
                    "num_objects": 2,
                    "bboxes": [[x,y,w,h], ...],
                },
                ...
            ],
            "total_frames": int,
            "positive_frames": int,  (움직임 있는 프레임)
            "negative_frames": int,  (움직임 없는 프레임)
            "total_annotations": int,
        }
    """
    print(f"[GT] JSON 파일 로딩: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    # 이미지 필터링
    images = data["images"]
    if sequence_range:
        start_id, end_id = sequence_range
        images = [img for img in images if start_id <= img["id"] <= end_id]
    print(f"[GT] 대상 이미지: {len(images)}장")

    # image_id → 어노테이션 매핑 (ignore=0, category_id != 4만)
    valid_anns = {}
    for ann in data["annotations"]:
        if ann.get("ignore", 0) == 1:
            continue
        if ann.get("category_id", 0) == 4:  # "ignore" 카테고리
            continue
        img_id = ann["image_id"]
        if sequence_range:
            if img_id < sequence_range[0] or img_id > sequence_range[1]:
                continue
        if img_id not in valid_anns:
            valid_anns[img_id] = []
        valid_anns[img_id].append(ann["bbox"])  # [x, y, w, h]

    # 프레임별 GT 생성
    frames = []
    for img in sorted(images, key=lambda x: x["id"]):
        img_id = img["id"]
        bboxes = valid_anns.get(img_id, [])
        frames.append({
            "image_id": img_id,
            "file_name": img["file_name"],
            "has_motion": len(bboxes) > 0,
            "num_objects": len(bboxes),
            "bboxes": bboxes,
        })

    positive = sum(1 for f in frames if f["has_motion"])
    total_anns = sum(f["num_objects"] for f in frames)

    result = {
        "frames": frames,
        "total_frames": len(frames),
        "positive_frames": positive,
        "negative_frames": len(frames) - positive,
        "total_annotations": total_anns,
    }

    print(f"[GT] 총 {result['total_frames']}프레임, "
          f"양성 {result['positive_frames']}({positive/len(frames)*100:.1f}%), "
          f"음성 {result['negative_frames']}, "
          f"어노테이션 {total_anns}개")

    return result


def parse_nightowls_vbb(vbb_path: str) -> Dict:
    """
    NightOwls VBB(MATLAB .mat) → 프레임별 ground truth 변환.

    Args:
        vbb_path: V001.vbb 등 경로

    Returns:
        parse_nightowls_json과 동일한 형식
    """
    import scipy.io as sio

    print(f"[GT] VBB 파일 로딩: {vbb_path}")
    data = sio.loadmat(vbb_path, squeeze_me=True, struct_as_record=False)
    A = data["A"]

    frames = []
    total_anns = 0

    for i in range(A.nFrame):
        obj = A.objLists[i]
        bboxes = []

        if obj is not None and (not isinstance(obj, np.ndarray) or obj.size > 0):
            items = obj if isinstance(obj, np.ndarray) and obj.ndim > 0 else [obj]
            for o in items:
                if not hasattr(o, "id"):
                    continue
                lbl_idx = int(o.id) - 1
                if lbl_idx < len(A.objLbl) and A.objLbl[lbl_idx] == "person":
                    x, y, w, h = [int(v) for v in o.pos]
                    bboxes.append([x, y, w, h])

        total_anns += len(bboxes)
        frames.append({
            "image_id": int(A.imageIds[i]) if hasattr(A, "imageIds") else i,
            "file_name": f"frame_{i:06d}.png",
            "has_motion": len(bboxes) > 0,
            "num_objects": len(bboxes),
            "bboxes": bboxes,
        })

    positive = sum(1 for f in frames if f["has_motion"])

    result = {
        "frames": frames,
        "total_frames": len(frames),
        "positive_frames": positive,
        "negative_frames": len(frames) - positive,
        "total_annotations": total_anns,
    }

    print(f"[GT] 총 {result['total_frames']}프레임, "
          f"양성 {result['positive_frames']}({positive/len(frames)*100:.1f}%), "
          f"음성 {result['negative_frames']}, "
          f"어노테이션 {total_anns}개")

    return result


# ── 시퀀스 ID 매핑 (V001~V006 = imageId 범위) ──

# V001.vbb 분석 결과: imageIds 7000000~7001317 (1318프레임)
# 나머지는 VBB 파일을 열어서 확인해야 함
SEQUENCE_RANGES = {
    "V001": (7000000, 7001317),
    # V002~V006은 VBB 파일의 imageIds로 확인 후 추가
}


def get_sequence_gt(json_path: str, sequence_name: str) -> Dict:
    """특정 시퀀스의 GT만 추출"""
    if sequence_name not in SEQUENCE_RANGES:
        raise ValueError(f"알 수 없는 시퀀스: {sequence_name}. "
                         f"사용 가능: {list(SEQUENCE_RANGES.keys())}")
    return parse_nightowls_json(json_path, SEQUENCE_RANGES[sequence_name])


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법:")
        print("  python parse_gt.py nightowls_validation.json")
        print("  python parse_gt.py V001.vbb")
        sys.exit(1)

    path = sys.argv[1]

    if path.endswith(".json"):
        # V001 범위만 테스트
        gt = parse_nightowls_json(path, sequence_range=(7000000, 7001317))
    elif path.endswith(".vbb"):
        gt = parse_nightowls_vbb(path)
    else:
        print("지원하지 않는 형식:", path)
        sys.exit(1)

    # 샘플 출력
    print(f"\n처음 5프레임:")
    for f in gt["frames"][:5]:
        print(f"  {f['image_id']}: motion={f['has_motion']}, objects={f['num_objects']}")