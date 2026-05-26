"""REST API 라우터 — 영상 업로드, 설정, 분석 실행, 평가 결과, 알림"""

import os
import uuid
import json
import cv2

from fastapi import APIRouter, UploadFile, File, HTTPException

from ..config import settings
from ..utils.video_io import get_video_info
from ..processing.pipeline import Pipeline


router = APIRouter(prefix="/api")

UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 알림 저장소 (메모리)
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


@router.get("/videos")
async def list_videos():
    """업로드된 영상 목록"""
    videos = []

    for f in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, f)

        try:
            info = get_video_info(path)
            vid = os.path.splitext(f)[0]

            videos.append({
                "video_id": vid,
                "filename": f,
                **info,
            })

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

    return {
        "ok": True,
        "settings": settings.model_dump(),
    }


@router.post("/analyze/{video_id}")
async def analyze_video(video_id: str):
    """업로드된 영상을 분석하고 결과를 results/{video_id}.json에 저장"""

    # 1. uploads 폴더에서 video_id에 해당하는 영상 찾기
    video_path = None

    for f in os.listdir(UPLOAD_DIR):
        if f.startswith(video_id + "."):
            video_path = os.path.join(UPLOAD_DIR, f)
            break

    if video_path is None:
        raise HTTPException(status_code=404, detail="영상 파일 없음")

    # 2. 영상 열기
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="영상 열기 실패")

    # 3. 파이프라인 생성
    pipeline = Pipeline(settings)
    pipeline.reset()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    analyzed_frames = 0
    motion_count = 0
    max_risk = 0.0
    risk_sum = 0.0
    detections = []

    # 초당 몇 프레임 분석할지 설정값 기준으로 조절
    checks_per_second = getattr(settings, "checks_per_second", 2)

    if fps and fps > 0:
        frame_interval = max(1, int(fps // checks_per_second))
    else:
        frame_interval = 1

    frame_index = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_index += 1

        # 모든 프레임을 다 분석하면 느리므로 일정 간격으로 분석
        if frame_index % frame_interval != 0:
            continue

        analyzed_frames += 1

        try:
            result = pipeline.process_frame(frame, include_previews=False)
        except Exception as e:
            cap.release()
            raise HTTPException(
                status_code=500,
                detail=f"{frame_index}번 프레임 분석 중 오류: {e}"
            )

        # frame_jpeg는 bytes라 JSON 저장 불가 → 제거
        result.pop("frame_jpeg", None)

        motion = result.get("motion", {})
        risk_score = float(motion.get("risk_score", 0.0))

        risk_sum += risk_score
        max_risk = max(max_risk, risk_score)

        if motion.get("detected"):
            motion_count += 1

            detection = {
                "frame_number": frame_index,
                "time": round(frame_index / fps, 2) if fps else 0,
                "risk_score": risk_score,
                "risk_level": motion.get("risk_level"),
                "boxes": motion.get("boxes", []),
                "total_motion_pixels": motion.get("total_motion_pixels", 0),
            }

            detections.append(detection)

            # 알림 저장
            alerts_store.append({
                "video_id": video_id,
                "frame_number": frame_index,
                "time": detection["time"],
                "risk_score": risk_score,
                "risk_level": motion.get("risk_level"),
                "message": "움직임 감지",
            })

    cap.release()

    avg_risk = risk_sum / analyzed_frames if analyzed_frames > 0 else 0.0

    # 4. 결과 JSON 생성
    result_data = {
        "video_id": video_id,
        "video_info": {
            "path": video_path,
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
        },
        "summary": {
            "analyzed_frames": analyzed_frames,
            "motion_count": motion_count,
            "avg_risk": round(avg_risk, 2),
            "max_risk": round(max_risk, 2),
            "alert_count": len(detections),
        },
        "detections": detections,
        "heatmap_base64": pipeline.get_heatmap_base64(),
    }

    # 5. results/{video_id}.json 저장
    result_path = os.path.join(RESULTS_DIR, f"{video_id}.json")

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    return {
        "ok": True,
        "message": "분석 완료",
        "video_id": video_id,
        "result_path": result_path,
        "summary": result_data["summary"],
    }


@router.get("/metrics/{video_id}")
async def get_metrics(video_id: str):
    """분석 결과/평가 지표 조회"""
    result_path = os.path.join(RESULTS_DIR, f"{video_id}.json")

    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="분석 결과 없음")

    with open(result_path, "r", encoding="utf-8") as f:
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