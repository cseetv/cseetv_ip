# cseetv_ip — 영상처리 백엔드

CCTV 이상 움직임 감지 및 알림 시스템의 영상처리 서버.

## 기술 스택
- FastAPI + uvicorn
- OpenCV (headless) + NumPy
- WebSocket (실시간 프레임 스트리밍)

## 로컬 실행

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## API

- `GET /health` — 서버 상태 확인
- `POST /api/upload` — 영상 업로드
- `GET /api/settings` — 설정 조회
- `POST /api/settings` — 설정 변경
- `GET /api/metrics/{video_id}` — 평가 결과
- `WS /ws/stream` — 실시간 영상 처리

## 배포 (Render)

Docker 기반. `Dockerfile` 포함.

## 프론트엔드 연동

- cseetv_fe (React): https://github.com/cseetv/cseetv_fe
