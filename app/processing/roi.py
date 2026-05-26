"""ROI 마스킹 모듈 — 다각형 관심 영역 설정

수업 연결: 토픽 #14 Masking
사용 이유: 관심 영역 외 움직임 무시. 가로등/나뭇잎/도로 차량 등 무관한 움직임 차단.
          일반적으로 FPR 60~80% 감소.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional


def create_roi_mask(
    frame_shape: tuple,
    polygons: List[Dict],
) -> Optional[np.ndarray]:
    """
    다각형 ROI 좌표 → 바이너리 마스크 생성.

    Args:
        frame_shape: (height, width) 또는 (height, width, channels)
        polygons: [
            { "id": "roi1", "name": "진열대", "points": [[x1,y1],[x2,y2],...] },
            ...
        ]

    Returns:
        mask: ROI 영역=255, 나머지=0. 다각형이 없으면 None (전체 통과).
    """
    if not polygons:
        return None

    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for poly in polygons:
        pts = poly.get("points", [])
        if len(pts) < 3:
            continue
        np_pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [np_pts], 255)

    return mask


def apply_roi(
    motion_mask: np.ndarray,
    roi_mask: Optional[np.ndarray],
) -> np.ndarray:
    """
    움직임 마스크에 ROI 적용 → ROI 안의 움직임만 남김.

    Args:
        motion_mask: 움직임 이진 마스크 (0/255)
        roi_mask: ROI 마스크 (None이면 전체 통과)

    Returns:
        masked: ROI 적용된 움직임 마스크
    """
    if roi_mask is None:
        return motion_mask
    return cv2.bitwise_and(motion_mask, motion_mask, mask=roi_mask)


def check_boxes_in_roi(
    boxes: List[Dict],
    polygons: List[Dict],
) -> List[Dict]:
    """
    각 감지 박스가 어떤 ROI 안에 있는지 확인하여 roi_name 추가.
    """
    if not polygons:
        return boxes

    for box in boxes:
        cx = box["x"] + box["w"] // 2
        cy = box["y"] + box["h"] // 2

        box["in_roi"] = None
        for poly in polygons:
            pts = np.array(poly.get("points", []), dtype=np.int32)
            if len(pts) < 3:
                continue
            result = cv2.pointPolygonTest(pts, (float(cx), float(cy)), False)
            if result >= 0:
                box["in_roi"] = poly.get("name", poly.get("id", "unknown"))
                break

    return boxes
