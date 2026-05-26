"""cseetv 백엔드 — FastAPI 앱 진입점"""

import os
import json
import traceback
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="cseetv API",
    description="CCTV 이상 움직임 감지 및 알림 시스템 백엔드",
    version="1.0.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── import 시도 + 에러 기록 ──
_import_error = None
_ws_handler = None
_api_router = None

try:
    from app.api.websocket import websocket_endpoint
    from app.api.routes import router as api_router
    _ws_handler = websocket_endpoint
    _api_router = api_router
    print("✅ 모든 모듈 import 성공")
except Exception as e:
    _import_error = traceback.format_exc()
    print(f"❌ 모듈 import 실패:\n{_import_error}")


# ── REST 라우터 ──
if _api_router:
    app.include_router(_api_router)


# ── Health (import 에러도 여기서 확인 가능) ──
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "cseetv-ip",
        "modules_loaded": _import_error is None,
        "import_error": _import_error,
    }


# ── WebSocket ──
@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    # 모듈 로드 성공 시 → 실제 핸들러
    if _ws_handler:
        await _ws_handler(websocket)
        return

    # 모듈 로드 실패 시 → 에러 메시지 전송 후 에코 모드
    await websocket.accept()
    try:
        await websocket.send_json({
            "type": "error",
            "message": f"서버 모듈 로드 실패: {_import_error}",
        })

        # 에코 모드 (프론트가 최소한 연결은 됨)
        while True:
            data = await websocket.receive()

            if "bytes" in data and data["bytes"]:
                await websocket.send_json({
                    "type": "frame_meta",
                    "frame_number": 0,
                    "motion": {
                        "detected": False, "boxes": [],
                        "total_motion_pixels": 0,
                        "risk_score": 0, "risk_level": "safe",
                    },
                    "quality": {
                        "brightness_mean": 0, "brightness_std": 0,
                        "entropy": 0, "histogram": [],
                        "diagnosis": "good",
                    },
                    "pipeline": {"steps_applied": ["fallback_echo"]},
                })
                await websocket.send_bytes(data["bytes"])

            elif "text" in data and data["text"]:
                msg = json.loads(data["text"])
                await websocket.send_json({"type": "echo", "received": msg})

    except Exception:
        pass