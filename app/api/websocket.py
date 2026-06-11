"""WebSocket 실시간 스트리밍 — 영상/카메라 처리의 핵심

통신 프로토콜:
  클라이언트 → 서버: JSON 제어 메시지 또는 바이너리 프레임
  서버 → 클라이언트: JSON 메타데이터 + 바이너리 JPEG 프레임
"""

import os
import json
import asyncio
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from ..config import settings as global_settings, ProcessingSettings
from ..processing.pipeline import Pipeline
from ..utils.video_io import iterate_frames, count_total_analysis_frames, get_video_info
from ..utils.encode import bytes_to_frame, frame_to_base64
from .routes import alerts_store

UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"


async def websocket_endpoint(websocket: WebSocket):
    """메인 WebSocket 핸들러"""
    await websocket.accept()

    # 연결별 설정 및 파이프라인
    local_settings = ProcessingSettings(**global_settings.model_dump())
    pipeline = Pipeline(local_settings)
    running = False

    try:
        while True:
            data = await websocket.receive()

            # 바이너리 데이터 = 카메라 프레임 (JPEG)
            if "bytes" in data and data["bytes"]:
                frame = bytes_to_frame(data["bytes"])
                if frame is not None:
                    result = pipeline.process_frame(frame)
                    await _send_result(websocket, result, local_settings)
                continue

            # 텍스트 데이터 = JSON 제어 메시지
            if "text" in data and data["text"]:
                msg = json.loads(data["text"])
                msg_type = msg.get("type", "")

                if msg_type == "start_video":
                    video_id = msg.get("video_id", "")
                    include_previews = msg.get("include_previews", False)
                    running = True
                    pipeline.reset()

                    # 비동기로 영상 처리 시작
                    asyncio.ensure_future(
                        _process_video(
                            websocket, pipeline, video_id,
                            local_settings, include_previews
                        )
                    )

                elif msg_type == "start_ip_camera":
                    camera_url = msg.get("url", "")
                    running = True
                    pipeline.reset()
                    asyncio.ensure_future(
                        _process_ip_camera(
                            websocket, pipeline, camera_url, local_settings
                        )
                    )

                elif msg_type == "camera_frame":
                    # base64로 전송된 카메라 프레임
                    import base64
                    frame_b64 = msg.get("frame", "")
                    if frame_b64:
                        from ..utils.encode import base64_to_frame
                        frame = base64_to_frame(frame_b64)
                        if frame is not None:
                            result = pipeline.process_frame(frame)
                            await _send_result(websocket, result, local_settings)

                elif msg_type == "update_settings":
                    new_vals = {k: v for k, v in msg.items() if k != "type"}
                    pipeline.update_settings(new_vals)
                    for k, v in new_vals.items():
                        if hasattr(local_settings, k):
                            setattr(local_settings, k, v)
                    await websocket.send_json({"type": "settings_updated", "ok": True})

                elif msg_type == "update_roi":
                    polygons = msg.get("polygons", [])
                    pipeline.update_roi(polygons)
                    await websocket.send_json({"type": "roi_updated", "ok": True})

                elif msg_type == "stop":
                    running = False
                    # 히트맵 전송
                    heatmap = pipeline.get_heatmap_base64()
                    if heatmap:
                        await websocket.send_json({
                            "type": "heatmap",
                            "heatmap_base64": heatmap,
                        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


async def _process_video(
    websocket: WebSocket,
    pipeline: Pipeline,
    video_id: str,
    settings: ProcessingSettings,
    include_previews: bool = False,
):
    """영상 파일을 프레임 단위로 처리하여 결과를 실시간 전송"""
    # 파일 찾기
    video_path = None
    for f in os.listdir(UPLOAD_DIR):
        if f.startswith(video_id):
            video_path = os.path.join(UPLOAD_DIR, f)
            break

    if not video_path:
        await websocket.send_json({"type": "error", "message": f"영상 {video_id}를 찾을 수 없습니다"})
        return

    total_frames = count_total_analysis_frames(video_path, settings.checks_per_second)
    analyzed = 0
    total_detections = 0
    all_risks = []

    try:
        for frame_num, timestamp, frame in iterate_frames(video_path, settings.checks_per_second):
            analyzed += 1

            # 프레임 처리
            send_previews = include_previews and analyzed <= 3  # 처음 3프레임만 미리보기
            result = pipeline.process_frame(frame, include_previews=send_previews)
            result["timestamp"] = round(timestamp, 2)

            # 결과 전송
            await _send_result(websocket, result, settings)

            # 통계 수집
            motion = result.get("motion", {})
            if motion.get("detected"):
                total_detections += 1
                _add_alert(result, settings)

            all_risks.append(motion.get("risk_score", 0))

            # 진행률 전송 (10프레임마다)
            if analyzed % 10 == 0:
                await websocket.send_json({
                    "type": "progress",
                    "current": analyzed,
                    "total": total_frames,
                })

            # CPU 양보 (비동기 루프 블로킹 방지)
            await asyncio.sleep(0.01)

    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        return

    # 완료
    avg_risk = sum(all_risks) / len(all_risks) if all_risks else 0
    max_risk = max(all_risks) if all_risks else 0

    summary = {
        "type": "done",
        "total_frames": total_frames,
        "analyzed_frames": analyzed,
        "total_detections": total_detections,
        "summary": {
            "avg_risk": round(avg_risk, 1),
            "max_risk": round(max_risk, 1),
            "alert_count": sum(1 for r in all_risks if r > settings.alert_threshold),
        },
    }

    # 히트맵
    heatmap = pipeline.get_heatmap_base64()
    if heatmap:
        summary["heatmap_base64"] = heatmap

    # 결과 저장
    result_path = os.path.join(RESULTS_DIR, f"{video_id}.json")
    with open(result_path, "w") as f:
        json.dump(summary["summary"], f)

    await websocket.send_json(summary)


async def _process_ip_camera(
    websocket: WebSocket,
    pipeline: Pipeline,
    camera_url: str,
    settings: ProcessingSettings,
):
    """IP 카메라 스트림 처리"""
    import cv2

    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        await websocket.send_json({"type": "error", "message": f"카메라 연결 실패: {camera_url}"})
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    check_interval = max(1, int(fps / settings.checks_per_second))
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await websocket.send_json({"type": "error", "message": "카메라 프레임 읽기 실패"})
                break

            frame_count += 1
            if frame_count % check_interval != 0:
                continue

            result = pipeline.process_frame(frame)
            await _send_result(websocket, result, settings)

            motion = result.get("motion", {})
            if motion.get("detected"):
                _add_alert(result, settings)

            await asyncio.sleep(0.01)

    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        cap.release()
        await websocket.send_json({"type": "done", "message": "카메라 스트림 종료"})


async def _send_result(websocket: WebSocket, result: dict, settings: ProcessingSettings):
    """처리 결과를 WebSocket으로 전송 (바이너리 or base64)"""
    frame_jpeg = result.pop("frame_jpeg", None)

    # 변화 없으면 이미지 생략 옵션
    motion = result.get("motion", {})
    skip_image = (
        settings.skip_unchanged_frames
        and not motion.get("detected", False)
        and result.get("frame_number", 0) % 10 != 0  # 10프레임마다는 전송
    )

    if settings.transfer_mode == "binary" and frame_jpeg and not skip_image:
        # 메타데이터 먼저, 바이너리 이미지 다음
        meta = {"type": "frame_meta", **result}

        # enhanced_previews에서 base64 이미지는 이미 문자열이므로 JSON 가능
        await websocket.send_json(meta)
        await websocket.send_bytes(frame_jpeg)
    else:
        # base64 모드 또는 이미지 생략
        meta = {"type": "frame_result", **result}
        if frame_jpeg and not skip_image:
            import base64
            meta["frame_base64"] = base64.b64encode(frame_jpeg).decode("utf-8")
        else:
            meta["frame_skipped"] = True
        await websocket.send_json(meta)


def _add_alert(result: dict, settings: ProcessingSettings):
    """알림 저장소에 추가"""
    motion = result.get("motion", {})
    risk = motion.get("risk_score", 0)

    if risk < settings.alert_threshold:
        return

    alert = {
        "timestamp": datetime.now().isoformat(),
        "risk_score": risk,
        "risk_level": motion.get("risk_level", "safe"),
        "motion_pixels": motion.get("total_motion_pixels", 0),
        "boxes": motion.get("boxes", []),
        "message": f"움직임 감지 (위험도 {risk:.0f})",
    }
    alerts_store.append(alert)

    # 최대 500건 유지
    if len(alerts_store) > 500:
        alerts_store.pop(0)
