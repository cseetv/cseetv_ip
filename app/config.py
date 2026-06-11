"""cseetv 설정값 관리"""

from pydantic import BaseModel


class ProcessingSettings(BaseModel):
    """영상처리 파이프라인 설정 (프론트에서 실시간 변경 가능)"""

    # 밝기 보정
    use_clahe: bool = True
    target_brightness: int = 120
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8

    # 노이즈 제거
    use_fastNlMeans: bool = False       # 실험 결과: 효과 없음, 140배 느림
    denoise_h: int = 7
    use_gaussian: bool = True
    gaussian_kernel: int = 5
    use_median: bool = False             # 실험 결과: Gaussian만으로 충분
    median_kernel: int = 5

    # Image Averaging
    use_averaging: bool = False          # Running Average로 대체
    averaging_n: int = 5

    # 움직임 감지 방식
    detection_method: str = "running_avg"  # "frame_diff" | "running_avg" | "mog2"
    running_avg_alpha: float = 0.05

    # 임계값
    threshold_value: int = 20            # 실험 결과: T=20 최적
    use_adaptive_threshold: bool = False
    use_otsu: bool = False
    min_motion_area: int = 200
    morph_kernel_size: int = 5
    dilate_iterations: int = 2

    # Shadow Removal
    use_shadow_removal: bool = False     # 실험 결과: 환경 의존적, 기본 OFF
    shadow_h_thresh: int = 15
    shadow_s_thresh: int = 30

    # Temporal Smoothing
    use_temporal_smoothing: bool = False
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
    transfer_mode: str = "binary"
    skip_unchanged_frames: bool = True

    # 알림
    alert_threshold: float = 60.0

    # 단계별 미리보기
    include_step_previews: bool = True


# 전역 설정 인스턴스
settings = ProcessingSettings()


# CORS 허용 도메인
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://cseetv-fe.vercel.app",
]