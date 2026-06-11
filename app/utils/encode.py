"""OpenCV 프레임 ↔ 전송 형식 변환"""

import base64
import cv2
import numpy as np


def frame_to_jpeg(frame: np.ndarray, quality: int = 70) -> bytes:
    """OpenCV BGR 프레임 → JPEG 바이트"""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def frame_to_base64(frame: np.ndarray, quality: int = 70) -> str:
    """OpenCV BGR 프레임 → base64 문자열"""
    jpeg = frame_to_jpeg(frame, quality)
    return base64.b64encode(jpeg).decode("utf-8")


def base64_to_frame(b64: str) -> np.ndarray:
    """base64 문자열 → OpenCV BGR 프레임"""
    jpeg = base64.b64decode(b64)
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def bytes_to_frame(data: bytes) -> np.ndarray:
    """JPEG 바이트 → OpenCV BGR 프레임"""
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def gray_to_jpeg(gray: np.ndarray, quality: int = 70) -> bytes:
    """Grayscale 프레임 → JPEG 바이트"""
    _, buf = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def gray_to_base64(gray: np.ndarray, quality: int = 70) -> str:
    """Grayscale 프레임 → base64 문자열"""
    return base64.b64encode(gray_to_jpeg(gray, quality)).decode("utf-8")
