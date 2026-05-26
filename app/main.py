"""cseetv 백엔드 — FastAPI 앱 진입점"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import router as api_router
from .api.websocket import websocket_endpoint
from .config import ALLOWED_ORIGINS

app = FastAPI(
    title="cseetv API",
    description="CCTV 이상 움직임 감지 및 알림 시스템 백엔드",
    version="1.0.0",
)

# CORS 설정
origins = os.environ.get("ALLOWED_ORIGINS", "").split(",")
origins = [o.strip() for o in origins if o.strip()] or ALLOWED_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST 라우터
app.include_router(api_router)


# WebSocket
@app.websocket("/ws/stream")
async def ws_stream(websocket):
    await websocket_endpoint(websocket)


# Health Check
@app.get("/health")
async def health():
    return {"status": "ok", "service": "cseetv-ip"}
