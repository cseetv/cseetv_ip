"""영상처리 메인 파이프라인 — 전체 흐름 제어

파이프라인 순서:
  ① 밝기 보정 (CLAHE)
  ② 노이즈 제거 (fastNlMeans + Gaussian + Median)
  ③ Image Averaging (N프레임)
  ④ Grayscale 변환
  ⑤ Frame Difference
  ⑥ (Adaptive) Threshold + Morphology
  ⑦ Shadow Removal
  ⑧ ROI Masking
  ⑨ Contour + Area Filtering
  → 결과: motion_boxes, risk_score, stats
"""

import cv2
import numpy as np
from typing import Optional, Dict
from ..config import ProcessingSettings
from .brightness import adjust_brightness
from .denoising import denoise
from .averaging import FrameAverager
from .motion import detect_motion, TemporalSmoother
from .shadow import remove_shadows
from .roi import create_roi_mask, apply_roi, check_boxes_in_roi
from .enhancement import get_enhancement_steps
from ..utils.encode import frame_to_jpeg, frame_to_base64


class Pipeline:
    """영상처리 파이프라인 전체를 관리하는 클래스"""

    def __init__(self, settings: ProcessingSettings):
        self.settings = settings
        self.averager = FrameAverager(settings.averaging_n)
        self.smoother = TemporalSmoother(settings.temporal_frames)
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_bgr: Optional[np.ndarray] = None
        self.roi_mask: Optional[np.ndarray] = None
        self.heatmap: Optional[np.ndarray] = None
        self.frame_count = 0

    def update_settings(self, new_settings: dict):
        """프론트에서 실시간 설정 변경 시 호출"""
        for key, value in new_settings.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)

        # Averaging N 변경 반영
        self.averager.update_n(self.settings.averaging_n)
        self.smoother.required = self.settings.temporal_frames

    def update_roi(self, polygons: list):
        """ROI 다각형 업데이트. 다음 process_frame에서 마스크 재생성."""
        self.settings.roi_polygons = polygons
        self.roi_mask = None  # 재생성 트리거

    def reset(self):
        """새 영상 시작 시 상태 초기화"""
        self.prev_gray = None
        self.prev_bgr = None
        self.roi_mask = None
        self.heatmap = None
        self.frame_count = 0
        self.averager.reset()
        self.smoother.reset()

    def process_frame(
        self, frame: np.ndarray, include_previews: bool = False
    ) -> Dict:
        """
        프레임 1장을 파이프라인으로 처리.

        Args:
            frame: 원본 BGR 프레임
            include_previews: True면 보정 단계별 미리보기 포함 (전송량 증가)

        Returns:
            결과 dict (WebSocket으로 전송할 데이터)
        """
        s = self.settings
        self.frame_count += 1
        steps_applied = []

        # ⓪ 해상도 표준화 (모든 프레임을 동일 크기로)
        TARGET_W, TARGET_H = 640, 480
        h_orig, w_orig = frame.shape[:2]
        if w_orig != TARGET_W or h_orig != TARGET_H:
            frame = cv2.resize(frame, (TARGET_W, TARGET_H))

        # ① 밝기 보정
        adjusted, brightness_info = adjust_brightness(
            frame, s.target_brightness, s.clahe_clip_limit, s.clahe_tile_size
        )
        steps_applied.append("clahe")

        # ② 노이즈 제거
        denoised = denoise(
            adjusted, s.denoise_h, s.use_gaussian, s.gaussian_kernel,
            s.use_median, s.median_kernel
        )
        if s.denoise_h > 0:
            steps_applied.append("fastNlMeans")
        if s.use_gaussian:
            steps_applied.append("gaussianBlur")
        if s.use_median:
            steps_applied.append("medianFilter")

        # ③ Grayscale 변환
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

        # ④ Image Averaging
        if s.use_averaging:
            gray = self.averager.add_and_average(gray)
            steps_applied.append(f"averaging_n{s.averaging_n}")

        # 영상 품질 계산
        quality_stats = self._calc_quality(gray)
        quality_stats.update(brightness_info)

        # ⑤~⑨ 움직임 감지 (이전 프레임이 있을 때만)
        motion_result = {
            "detected": False,
            "boxes": [],
            "total_motion_pixels": 0,
            "risk_score": 0.0,
            "risk_level": "safe",
        }

        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            # 동적 임계값
            curr_brightness = quality_stats.get("brightness_after", 120.0)

            # Frame Difference + Morphology + Contour
            raw_motion = detect_motion(
                self.prev_gray, gray,
                threshold_value=s.threshold_value,
                use_adaptive=s.use_adaptive_threshold,
                use_dynamic=s.use_dynamic_threshold,
                dynamic_base=s.dynamic_base,
                dynamic_factor=s.dynamic_factor,
                brightness=curr_brightness,
                min_area=s.min_motion_area,
                morph_kernel_size=s.morph_kernel_size,
                dilate_iterations=s.dilate_iterations,
            )
            steps_applied.extend(["frameDiff", "morphOpen", "dilate", "contourFilter"])

            binary_mask = raw_motion["binary_mask"]

            # ⑦ Shadow Removal
            if s.use_shadow_removal and self.prev_bgr is not None:
                binary_mask = remove_shadows(
                    binary_mask, self.prev_bgr, denoised,
                    s.shadow_h_thresh, s.shadow_s_thresh
                )
                steps_applied.append("shadowRemoval")

            # ⑧ ROI Masking
            if s.roi_polygons and self.roi_mask is None:
                self.roi_mask = create_roi_mask(frame.shape, s.roi_polygons)

            if self.roi_mask is not None:
                binary_mask = apply_roi(binary_mask, self.roi_mask)
                steps_applied.append("roiMask")

            # Contour 재검출 (Shadow/ROI 적용 후)
            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            boxes = []
            for c in contours:
                area = cv2.contourArea(c)
                if area < s.min_motion_area:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area)})

            # ROI별 소속 확인
            if s.roi_polygons:
                boxes = check_boxes_in_roi(boxes, s.roi_polygons)

            motion_pixels = int(np.count_nonzero(binary_mask))
            total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
            risk_score = min(100.0, (motion_pixels / total_pixels) * 2500)

            detected = len(boxes) > 0

            # Temporal Smoothing
            if s.use_temporal_smoothing:
                detected = self.smoother.update(detected)
                steps_applied.append("temporalSmoothing")

            risk_level = "danger" if risk_score > 70 else "warn" if risk_score > 40 else "safe"

            motion_result = {
                "detected": detected,
                "boxes": boxes,
                "total_motion_pixels": motion_pixels,
                "risk_score": round(risk_score, 1),
                "risk_level": risk_level,
                "threshold_used": raw_motion.get("threshold_used", s.threshold_value),
            }

            # 히트맵 누적
            if self.heatmap is None:
                self.heatmap = np.zeros(binary_mask.shape, dtype=np.float32)
            self.heatmap += binary_mask.astype(np.float32) / 255.0

        # 결과 프레임 생성 (박스 그리기)
        result_frame = self._draw_boxes(denoised, motion_result["boxes"], motion_result)

        # 이전 프레임 저장
        self.prev_gray = gray.copy()
        self.prev_bgr = denoised.copy()

        # 결과 조합
        output = {
            "frame_number": self.frame_count,
            "motion": motion_result,
            "quality": quality_stats,
            "pipeline": {"steps_applied": steps_applied},
        }

        # 프레임 이미지 (JPEG 바이트)
        output["frame_jpeg"] = frame_to_jpeg(result_frame, s.jpeg_quality)

        # 보정 미리보기 (요청 시에만)
        if include_previews:
            output["pipeline"]["enhanced_previews"] = get_enhancement_steps(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), quality=50
            )

        return output

    def _calc_quality(self, gray: np.ndarray) -> dict:
        """영상 품질 지표 계산"""
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))

        # 히스토그램
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().tolist()

        # 엔트로피
        total = gray.size
        probs = [h / total for h in hist if h > 0]
        entropy = -sum(p * np.log2(p) for p in probs) if probs else 0.0

        # 진단
        left_ratio = sum(hist[:85]) / total
        right_ratio = sum(hist[170:]) / total

        if left_ratio > 0.6:
            diagnosis = "under_exposed"
        elif right_ratio > 0.6:
            diagnosis = "over_exposed"
        elif std_val < 30:
            diagnosis = "low_contrast"
        else:
            diagnosis = "good"

        return {
            "brightness_mean": round(mean_val, 1),
            "brightness_std": round(std_val, 1),
            "entropy": round(entropy, 2),
            "histogram": hist,
            "diagnosis": diagnosis,
        }

    def _draw_boxes(self, frame: np.ndarray, boxes: list, motion: dict) -> np.ndarray:
        """결과 프레임에 감지 박스와 상태 정보 그리기"""
        result = frame.copy()

        for box in boxes:
            color = (0, 0, 255)  # 빨간 박스
            cv2.rectangle(
                result,
                (box["x"], box["y"]),
                (box["x"] + box["w"], box["y"] + box["h"]),
                color, 2
            )
            # ROI 이름 표시
            roi_name = box.get("in_roi")
            if roi_name:
                cv2.putText(
                    result, roi_name,
                    (box["x"], box["y"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                )

        # 상태 텍스트
        if motion.get("detected"):
            cv2.putText(result, "MOTION DETECTED", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(result, "Normal", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        risk = motion.get("risk_score", 0)
        cv2.putText(result, f"Risk: {risk:.0f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return result

    def get_heatmap_base64(self, quality: int = 70) -> Optional[str]:
        """누적 히트맵을 컬러맵으로 변환하여 base64 반환"""
        if self.heatmap is None:
            return None
        normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
        return frame_to_base64(colored, quality)