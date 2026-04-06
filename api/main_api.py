import os
from pathlib import Path
from fastapi import FastAPI ,  APIRouter, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse , StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from Model.detector import VehicleDetector

import asyncio
import base64
import tempfile
import time
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_VIDEO_MB = 500
router = APIRouter()
MODEL_PATH     = r"F:\Omar 3amora\حنكشة projects\TrafficGuard_ComputerVision_ETE_Proj\dataset\runs\detect\train5\weights\best.pt"
CONF_THRESHOLD = 0.5
UI = Path(r"F:\Omar 3amora\حنكشة projects\TrafficGuard_ComputerVision_ETE_Proj\UI")

_detector: VehicleDetector | None = None
# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health")
async def health():
    return {
        "status": "ok" if Path(MODEL_PATH).exists() else "model_missing",
        "model_path": MODEL_PATH,
        "conf_threshold": CONF_THRESHOLD,
    }

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

@router.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...), conf: float = Query(default=None)):
    detector = get_detector()
    if conf is not None:
        detector.conf_threshold = conf

    raw = await file.read()
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Cannot decode image.")

    result = detector.detect(frame)
    jpg = detector.encode(result["frame"], quality=90)

    return JSONResponse({
        "image_base64": base64.b64encode(jpg).decode(),
        "detections": result["detections"],
        "counts": result["counts"],
        "total": result["total"],
        "inference_ms": result["inference_ms"],
    })


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

@router.post("/api/detect/video")
async def detect_video(file: UploadFile = File(...)):
    detector = get_detector()

    size_bytes = 0
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix or ".mp4")

    try:
        while chunk := await file.read(1024 * 1024):
            size_bytes += len(chunk)
            if size_bytes > MAX_VIDEO_MB * 1024 * 1024:
                tmp.close()
                raise HTTPException(413, f"Video exceeds {MAX_VIDEO_MB} MB limit.")
            tmp.write(chunk)
        tmp.flush()
        tmp_path = tmp.name
    finally:
        tmp.close()

    def generate():
        cap = cv2.VideoCapture(tmp_path)
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                jpg = detector.encode(detector.detect(frame)["frame"])
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        finally:
            cap.release()
            os.unlink(tmp_path)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------------------
# WebSocket – Webcam
# ---------------------------------------------------------------------------

@router.websocket("/ws/webcam")
async def webcam_ws(websocket: WebSocket):
    await websocket.accept()
    detector = get_detector()

    try:
        while True:
            frame = cv2.imdecode(np.frombuffer(await websocket.receive_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            t0 = time.perf_counter()
            result = detector.detect(frame)
            wall_ms = (time.perf_counter() - t0) * 1000

            await websocket.send_json({
                "image_base64": base64.b64encode(detector.encode(result["frame"], quality=82)).decode(),
                "counts": result["counts"],
                "total": result["total"],
                "inference_ms": result["inference_ms"],
                "fps": round(1000 / max(wall_ms, 1), 1),
            })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[webcam ws] {e}")


# ---------------------------------------------------------------------------
# WebSocket – RTSP
# ---------------------------------------------------------------------------

@router.websocket("/ws/rtsp")
async def rtsp_ws(websocket: WebSocket):
    await websocket.accept()
    detector = get_detector()
    cap = None

    try:
        cap = cv2.VideoCapture(await websocket.receive_text())
        if not cap.isOpened():
            await websocket.send_json({"error": "Cannot open stream."})
            return

        delay = 1 / (cap.get(cv2.CAP_PROP_FPS) or 25)

        while True:
            ret, frame = cap.read()
            if not ret:
                await websocket.send_json({"error": "Stream ended or dropped."})
                break

            t0 = time.perf_counter()
            result = detector.detect(frame)
            wall_ms = (time.perf_counter() - t0) * 1000

            await websocket.send_json({
                "image_base64": base64.b64encode(detector.encode(result["frame"], quality=82)).decode(),
                "detections": result["detections"],   # now includes speed
                "counts": result["counts"],
                "total": result["total"],
                "inference_ms": result["inference_ms"],
                "fps": round(1000 / max(wall_ms, 1), 1),
            })
            await asyncio.sleep(max(0, delay - wall_ms / 1000))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[rtsp ws] {e}")
    finally:
        if cap is not None and cap.isOpened():
            cap.release()



def get_detector() -> VehicleDetector:
    global _detector
    if _detector is None:
        if not Path(MODEL_PATH).exists():
            raise RuntimeError(
                f"Model not found at '{MODEL_PATH}'. "
                "Place your best.pt in the /models directory."
            )
        _detector = VehicleDetector(MODEL_PATH, CONF_THRESHOLD)
    return _detector


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:

    app = FastAPI(
        title="Vehicle Detection API",
        version="1.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if UI.exists():
        app.mount("/static",
                  StaticFiles(directory=str(UI)),
                  name="static")

    @app.get("/", include_in_schema=False)
    async def index():

        html_file = UI / "index.html"

        if html_file.exists():
            return FileResponse(str(html_file))

        return JSONResponse({
            "message": "Vehicle Detection API — frontend not found"
        })

    app.include_router(router)

    return app