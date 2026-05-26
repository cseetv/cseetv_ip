"""노이즈 제거 모듈 — fastNlMeans + GaussianBlur + MedianFilter

수업 연결: 토픽 #16 Moving Average, #20 Gaussian Filter
사용 이유:
  - fastNlMeans: CLAHE 보정 후 밝기 증폭으로 noise도 증폭. 구조 보존하며 제거.
  - GaussianBlur: 고주파 노이즈 smoothing. Ideal LPF 대비 ringing 없음.
  - MedianFilter: 소금-후추 노이즈에 Gaussian보다 효과적. 에지 보존력 우수.
"""

import cv2
import numpy as np


def denoise(
    frame: np.ndarray,
    denoise_h: int = 7,
    use_gaussian: bool = True,
    gaussian_kernel: int = 5,
    use_median: bool = True,
    median_kernel: int = 5,
) -> np.ndarray:
    """
    3단계 노이즈 제거 파이프라인.
    각 필터를 on/off 가능.

    1) fastNlMeansDenoisingColored — 구조 보존 노이즈 제거
    2) GaussianBlur — 고주파 smoothing
    3) MedianFilter — salt-and-pepper / hot pixel 제거
    """
    result = frame.copy()

    # 1. Non-local means denoising
    if denoise_h > 0:
        result = cv2.fastNlMeansDenoisingColored(
            result, None,
            h=denoise_h,
            hColor=denoise_h,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    # 2. Gaussian blur
    if use_gaussian and gaussian_kernel > 0:
        k = gaussian_kernel if gaussian_kernel % 2 == 1 else gaussian_kernel + 1
        result = cv2.GaussianBlur(result, (k, k), 0)

    # 3. Median filter
    if use_median and median_kernel > 0:
        k = median_kernel if median_kernel % 2 == 1 else median_kernel + 1
        result = cv2.medianBlur(result, k)

    return result
