# cseetv_ip — 백엔드

야간 CCTV 영상의 이상 움직임 감지 및 알림 시스템 (백엔드 + 실험)

## 팀원
- 이채원 (202210133) 
- 이예랑 (202310126)

## 기술 스택
- Python 3.12
- FastAPI + Uvicorn (웹 서버)
- OpenCV (영상처리)
- WebSocket (실시간 프레임 전송)
- NumPy, SciPy (수치 연산)

## 실행 방법

### 서버 실행
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 실험 실행
```bash
# CDnet tramStation 전체 (2,501프레임)
python -m experiments.run --source cdnet --data data/tramStation --mode all

# 추가 기법 + AI 비교만
python -m experiments.run --source cdnet --data data/tramStation --mode advanced

# PPT용 단계별 이미지
python save_steps.py

# 보고서용 그래프
python generate_graphs.py
```

## 프로젝트 구조

```
app/
 ├── main.py                    # FastAPI 앱 엔트리
 ├── config.py                  # 파이프라인 설정 (최적값 반영)
 ├── api/
 │   ├── routes.py              # REST API 라우트
 │   └── websocket.py           # WebSocket 핸들러
 ├── processing/                # ★ 영상처리 파이프라인 (실험과 공유)
 │   ├── pipeline.py            # 통합 파이프라인 (Frame Diff / Running Avg / MOG2)
 │   ├── brightness.py          # CLAHE 밝기 보정
 │   ├── denoising.py           # Gaussian + Median + fastNlMeans
 │   ├── motion.py              # Frame Difference + Contour 검출
 │   ├── shadow.py              # HSV 기반 Shadow Removal
 │   ├── roi.py                 # ROI 다각형 마스킹
 │   ├── averaging.py           # Image Averaging
 │   ├── enhancement.py         # 보정 단계 정보
 │   └── threshold.py           # 적응적 임계값
 ├── evaluation/
 │   ├── quality.py             # 영상 품질 (μ, σ, 엔트로피)
 │   └── detection.py           # 감지 성능 (P, R, F1)
 └── utils/
     ├── encode.py              # JPEG/Base64 인코딩
     └── video_io.py            # 영상 입출력

experiments/
 ├── run.py                     # ★ 통합 실험 실행기
 ├── loaders.py                 # 데이터 로더 (CDnet / 영상 파일)
 └── results/                   # 실험 결과 CSV + 그래프 PNG

data/
 └── tramStation/               # CDnet 데이터셋 (제출 시 샘플 20장만)
```

## 영상처리 파이프라인

```
입력 → ① CLAHE → ② Gaussian → ③ Running Average → ④ Morphology → ⑤ (Shadow) → ⑥ ROI → 감지 결과
```

| 단계 | 기법 | 수업 토픽 |
|------|------|----------|
| ① 밝기 보정 | CLAHE | Histogram EQ |
| ② 노이즈 제거 | Gaussian + Median | Gaussian Filter |
| ③ 움직임 감지 | Running Average / Frame Diff | Image Averaging / Image Diff |
| ④ 후처리 | Morphology Open + Dilate | 모폴로지 |
| ⑤ 그림자 제거 | HSV Shadow Removal | (환경 의존적) |
| ⑥ ROI | Polygon Masking | Masking |

## 최적 설정 (실험 결과)

```python
detection_method = "running_avg"  # Running Average 배경 모델
running_avg_alpha = 0.05
use_clahe = True
use_gaussian = True
use_median = False       # Gaussian만으로 충분
use_fastNlMeans = False  # 효과 없음, 140배 느림
morph_kernel_size = 5
threshold_value = 20     # 최적 T
use_shadow_removal = False  # tramStation에서 역효과
```

## 실험 결과 요약

| 기법 | Pixel F1 | ΔF1 |
|------|----------|-----|
| Baseline (Frame Diff) | 0.407 | — |
| + Morphology ★ | 0.471 | +0.069 |
| Running Average ★★ | 0.566 | +0.096 |
| MOG2 (AI) | 0.614 | +0.144 |

**전통 기법(Running Average)으로 AI(MOG2)의 92.1% 성능 달성**

## 참고문헌

1. N. Goyette et al., "changedetection.net: A new change detection benchmark dataset," CVPR Workshops, 2012.
2. Z. Zivkovic, "Improved Adaptive Gaussian Mixture Model for Background Subtraction," ICPR, 2004.
3. OpenCV documentation, https://docs.opencv.org/
