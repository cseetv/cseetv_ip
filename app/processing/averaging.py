"""Image Averaging 모듈 — N프레임 가중 평균으로 랜덤 노이즈 제거

수업 연결: 토픽 #11 Image Averaging
사용 이유: 랜덤 노이즈는 프레임마다 다르지만 배경은 동일.
          N프레임 평균 시 이론상 SNR √N배 향상.
주의: N이 너무 크면 실제 움직임도 평균되어 뭉개짐.
"""

import numpy as np
from collections import deque


class FrameAverager:
    """최근 N프레임의 가중 평균을 관리하는 링 버퍼"""

    def __init__(self, n: int = 5):
        self.n = max(1, n)
        self.buffer: deque = deque(maxlen=self.n)

    def update_n(self, n: int):
        """N 값 동적 변경"""
        new_n = max(1, n)
        if new_n != self.n:
            self.n = new_n
            new_buffer: deque = deque(maxlen=self.n)
            for item in self.buffer:
                new_buffer.append(item)
            self.buffer = new_buffer

    def add_and_average(self, gray: np.ndarray) -> np.ndarray:
        """
        새 프레임 추가 후 가중 평균 반환.
        최근 프레임일수록 가중치가 높은 지수 가중 평균 사용.
        """
        self.buffer.append(gray.astype(np.float32))

        if len(self.buffer) == 1:
            return gray.copy()

        # 지수 가중치: 최신 프레임 = 가중치 높음
        n = len(self.buffer)
        weights = np.array([0.5 ** (n - 1 - i) for i in range(n)], dtype=np.float32)
        weights /= weights.sum()

        averaged = np.zeros_like(self.buffer[0], dtype=np.float32)
        for w, frame in zip(weights, self.buffer):
            averaged += w * frame

        return np.clip(averaged, 0, 255).astype(np.uint8)

    def reset(self):
        """버퍼 초기화"""
        self.buffer.clear()
