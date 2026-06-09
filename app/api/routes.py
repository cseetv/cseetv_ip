"""REST API 라우터 — 영상 업로드, 설정, 평가 결과, 알림"""

import os
import uuid
import json
from typing import Any

from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from pywebpush import webpush, WebPushException

from ..config import settings, VAPID_PRIVATE_KEY, VAPID_PUBLIC_KEY, VAPID_SUBJECT
from ..utils.video_io import get_video_info

router = APIRouter(prefix="/api")

UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

push_subscriptions: list[dict[str, Any]] = []
alerts_store: list = []


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """영상 파일 업로드"""
    video_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename or "video.mp4")[1]
    save_path = os.path.join(UPLOAD_DIR, f"{video_id}{ext}")

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    try:
        info = get_video_info(save_path)
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(status_code=400, detail=f"영상 파일 오류: {e}")

    return {
        "video_id": video_id,
        "filename": file.filename,
        "path": save_path,
        **info,
    }


@router.post("/push/subscribe")
async def subscribe_push(subscription: dict[str, Any] = Body(...)):
    """웹 푸시 구독 정보 저장"""
    if not VAPID_PUBLIC_KEY or not VAPID_PRIVATE_KEY:
        raise HTTPException(status_code=500, detail="VAPID 키가 설정되지 않았습니다.")

    if subscription not in push_subscriptions:
        push_subscriptions.append(subscription)
    return {"ok": True, "vapidPublicKey": VAPID_PUBLIC_KEY}


@router.post("/push/send")
async def send_push(message: dict[str, Any] = Body(...)):
    """등록된 구독자에게 푸시 전송"""
    if not VAPID_PUBLIC_KEY or not VAPID_PRIVATE_KEY:
        raise HTTPException(status_code=500, detail="VAPID 키가 설정되지 않았습니다.")

    payload = {
        "title": message.get("title", "cseetv 알림"),
        "body": message.get("body", "위험이 감지되었습니다."),
    }

    results = []
    for sub in list(push_subscriptions):
        try:
            webpush(
                subscription_info=sub,
                data=json.dumps(payload),
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims={"sub": VAPID_SUBJECT},
            )
            results.append({"subscription": sub, "status": "ok"})
        except WebPushException as exc:
            results.append({"subscription": sub, "status": "error", "detail": str(exc)})
            if exc.response and exc.response.status_code == 410:
                push_subscriptions.remove(sub)
    return {"ok": True, "result": results}


@router.get("/push/public_key")
async def get_push_public_key():
    """웹 푸시 구독에 사용할 공개 VAPID 키 반환"""
    if not VAPID_PUBLIC_KEY:
        raise HTTPException(status_code=500, detail="VAPID 공개 키가 설정되지 않았습니다.")
    return {"vapidPublicKey": VAPID_PUBLIC_KEY}


@router.get("/videos")
async def list_videos():
    """업로드된 영상 목록"""
    videos = []
    for f in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, f)
        try:
            info = get_video_info(path)
            vid = f.split(".")[0]
            videos.append({"video_id": vid, "filename": f, **info})
        except Exception:
            continue
    return {"videos": videos}


@router.get("/settings")
async def get_settings():
    """현재 설정 조회"""
    return settings.model_dump()


@router.post("/settings")
async def update_settings(new_settings: dict):
    """설정 변경"""
    for key, value in new_settings.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    return {"ok": True, "settings": settings.model_dump()}


@router.get("/metrics/{video_id}")
async def get_metrics(video_id: str):
    """분석 결과/평가 지표 조회"""
    result_path = os.path.join(RESULTS_DIR, f"{video_id}.json")
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="분석 결과 없음")

    with open(result_path, "r") as f:
        return json.load(f)


@router.get("/alerts")
async def get_alerts():
    """최근 알림 목록"""
    return {"alerts": alerts_store[-100:]}


@router.post("/alerts/clear")
async def clear_alerts():
    """알림 초기화"""
    alerts_store.clear()
    return {"ok": True}
