from datetime import datetime
import os
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from camera_stream import (
    CAMERA_LOCATIONS, DEFAULT_CAMERA_LOC,
    scan_single_image, stream_manager, get_camera_node
)
from database import IST, db_manager

app = FastAPI(
    title="SENTINEL ANPR - Vehicle Intelligence & Route Tracking",
    description="Enterprise-grade Automatic Number Plate Recognition and Surveillance Dashboard",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>SENTINEL ANPR Backend Running. Please check static/index.html.</h1>"


@app.get("/api/status")
async def get_system_status():
    return {
        "status": "ONLINE",
        "vision_engine": "YOLOv8n + EasyOCR",
        "database_connected": db_manager.is_connected,
        "database_mode": "MongoDB Atlas" if db_manager.is_connected else "Local Resilient Store (Auto-Sync Ready)",
        "active_camera": stream_manager.active_camera_id,
        "mode": stream_manager.mode,
        "cameras": list(CAMERA_LOCATIONS.keys()),
        "time_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    }


@app.get("/api/stats")
async def get_analytics_stats():
    return db_manager.get_analytics()


@app.get("/api/cameras")
async def get_cameras():
    return db_manager.get_all_cameras()


@app.get("/api/recent")
async def get_recent_sightings(limit: int = Query(50, ge=1, le=200)):
    return db_manager.get_recent_detections(limit=limit)


@app.get("/api/search")
async def search_vehicle(plate: str = Query(..., description="Vehicle registration plate number")):
    if not plate or not plate.strip():
        return JSONResponse(status_code=400, content={"error": "Plate number is required"})
    result = db_manager.search_by_plate(plate.strip())
    return result


# ── Primary MJPEG stream (legacy, uses stream_manager) ──────────────────────
@app.get("/api/video_feed")
async def video_feed():
    return StreamingResponse(
        stream_manager.generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ── Per-camera MJPEG streams ─────────────────────────────────────────────────
@app.get("/api/feed/{cam_id}")
async def camera_feed(cam_id: str):
    """Returns an independent MJPEG stream for the given camera node."""
    if cam_id not in CAMERA_LOCATIONS:
        return JSONResponse(status_code=404, content={"error": f"Unknown camera: {cam_id}"})
    node = get_camera_node(cam_id)
    return StreamingResponse(
        node.generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/scan_image")
async def scan_image_upload(file: UploadFile = File(...), cam_id: str = Form("CAM_004")):
    try:
        contents = await file.read()
        res = scan_single_image(contents, cam_id=cam_id)
        return res
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to process image: {str(e)}"})


@app.post("/api/camera/mode")
async def set_camera_mode(mode: str = Form(...), cam_id: str = Form("CAM_004")):
    if mode in ["simulation", "webcam"]:
        stream_manager.set_mode(mode)
        if mode == "webcam":
            stream_manager.active_camera_id = cam_id
            node = get_camera_node(cam_id)
            node.mode = "webcam"
        else:
            for c_id in CAMERA_LOCATIONS:
                get_camera_node(c_id).mode = "simulation"
        return {"success": True, "mode": stream_manager.mode, "camera": cam_id}
    return JSONResponse(status_code=400, content={"error": "Invalid mode. Use 'simulation' or 'webcam'"})


@app.post("/api/camera/switch")
async def switch_camera(
    cam_id: str = Form("CAM_004"),
    camera_index: int = Form(1),
    mode: str = Form("simulation")
):
    stream_manager.active_camera_id = cam_id
    stream_manager.camera_index = camera_index
    stream_manager.set_mode(mode)
    stream_manager._init_camera()
    return {
        "success": True,
        "active_camera": stream_manager.active_camera_id,
        "camera_index": stream_manager.camera_index,
        "mode": stream_manager.mode
    }


@app.post("/api/test_detection")
async def inject_test_detection(
    plate: str = Form("HR06A1245"),
    cam_id: str = Form("CAM_004"),
    direction: str = Form("Southbound")
):
    cam_info = CAMERA_LOCATIONS.get(cam_id, DEFAULT_CAMERA_LOC)
    now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    rec = db_manager.add_detection(
        cam_id, plate,
        location=cam_info["location"], latitude=cam_info["latitude"],
        longitude=cam_info["longitude"], direction=direction,
        timestamp=now_ist, confidence=0.965, status="high_confidence"
    )
    return {"success": True, "record": rec}


if __name__ == "__main__":
    print("=" * 65)
    print("  SENTINEL ANPR - Vehicle Intelligence & Route Tracking")
    print("  Command Center URL: http://localhost:8000")
    print("  API Documentation:  http://localhost:8000/docs")
    print("=" * 65)
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
