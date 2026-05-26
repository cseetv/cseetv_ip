"""cseetv 백엔드 — FastAPI 앱 진입점"""

import os
import json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="cseetv API",
    description="CCTV 이상 움직임 감지 및 알림 시스템 백엔드",
    version="1.0.0",
)

# CORS 설정
origins = os.environ.get("ALLOWED_ORIGINS", "").split(",")
origins = [o.strip() for o in origins if o.strip()] or [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://cseetv-fe.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 디버깅용 - 나중에 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health Check
@app.get("/health")
async def health():
    return {"status": "ok", "service": "cseetv-ip"}


# ── WebSocket 테스트 (import 없이 직접 구현) ──
@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive()

            # 바이너리 (카메라 프레임)
            if "bytes" in data and data["bytes"]:
                # 일단 에코: 받은 이미지를 그대로 돌려보냄
                await websocket.send_json({
                    "type": "frame_meta",
                    "frame_number": 1,
                    "motion": {
                        "detected": False,
                        "boxes": [],
                        "total_motion_pixels": 0,
                        "risk_score": 0,
                        "risk_level": "safe",
                    },
                    "quality": {
                        "brightness_mean": 120,
                        "brightness_std": 40,
                        "entropy": 6.5,
                        "histogram": [0] * 256,
                        "diagnosis": "good",
                    },
                    "pipeline": {"steps_applied": ["echo_test"]},
                })
                await websocket.send_bytes(data["bytes"])

            # JSON 메시지
            elif "text" in data and data["text"]:
                msg = json.loads(data["text"])
                await websocket.send_json({
                    "type": "echo",
                    "received": msg,
                    "message": "WebSocket 연결 성공!",
                })

    except Exception as e:
        print(f"WebSocket error: {e}")


# ── REST API (나중에 routes.py로 분리) ──
@app.post("/api/upload")
async def upload_placeholder():
    return {"message": "아직 미구현"}

@app.get("/api/settings")
async def get_settings():
    return {"threshold_value": 25, "min_motion_area": 200}