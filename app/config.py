"""cseetv 설정값 관리"""

import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class ProcessingSettings(BaseModel):
    """영상처리 파이프라인 설정 (프론트에서 실시간 변경 가능)"""

    # 밝기 보정
    target_brightness: int = 120
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8

    # 노이즈 제거
    denoise_h: int = 7
    use_gaussian: bool = True
    gaussian_kernel: int = 5
    use_median: bool = True
    median_kernel: int = 5

    # Image Averaging
    use_averaging: bool = True
    averaging_n: int = 5

    # 움직임 감지
    threshold_value: int = 25
    use_adaptive_threshold: bool = False
    min_motion_area: int = 200
    morph_kernel_size: int = 5
    dilate_iterations: int = 2

    # Shadow Removal
    use_shadow_removal: bool = True
    shadow_h_thresh: int = 15
    shadow_s_thresh: int = 30

    # Temporal Smoothing
    use_temporal_smoothing: bool = True
    temporal_frames: int = 3

    # Dynamic Threshold
    use_dynamic_threshold: bool = False
    dynamic_base: int = 25
    dynamic_factor: float = 0.1

    # ROI
    roi_polygons: list = []

    # 전송
    checks_per_second: int = 2
    jpeg_quality: int = 70
    transfer_mode: str = "binary"  # "binary" | "base64"
    skip_unchanged_frames: bool = True

    # 알림
    alert_threshold: float = 60.0


# 전역 설정 인스턴스
settings = ProcessingSettings()


# CORS 허용 도메인
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://cseetv-fe.vercel.app",
]

VAPID_PUBLIC_KEY = os.getenv("VAPID_PUBLIC_KEY", "")
VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY", "")
VAPID_SUBJECT = os.getenv("VAPID_SUBJECT", "mailto:admin@example.com")
