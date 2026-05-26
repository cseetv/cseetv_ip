"""영상 파일 읽기/프레임 추출"""

import cv2
import os
from typing import Generator, Tuple
import numpy as np


def get_video_info(path: str) -> dict:
    """영상 파일 메타데이터 조회"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {path}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": 0.0,
    }
    fps = info["fps"] if info["fps"] > 0 else 30
    info["duration"] = info["total_frames"] / fps
    cap.release()
    return info


def iterate_frames(
    path: str, checks_per_second: int = 2
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """
    영상 파일에서 프레임을 일정 간격으로 추출하는 제너레이터.

    yields: (frame_number, timestamp_sec, frame_bgr)
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    check_interval = max(1, int(fps / checks_per_second))
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % check_interval != 0:
                continue

            timestamp = frame_count / fps
            yield frame_count, timestamp, frame
    finally:
        cap.release()


def count_total_analysis_frames(path: str, checks_per_second: int = 2) -> int:
    """분석할 총 프레임 수 계산"""
    info = get_video_info(path)
    fps = info["fps"] if info["fps"] > 0 else 30
    check_interval = max(1, int(fps / checks_per_second))
    return info["total_frames"] // check_interval
