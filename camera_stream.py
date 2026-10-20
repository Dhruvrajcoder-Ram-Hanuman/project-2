import base64
from datetime import datetime
import io
import json
import os
import re
import threading
import time
import math
import cv2
import numpy as np

from database import IST, db_manager

# Lazy-loaded globals for YOLO and EasyOCR
_yolo_model = None
_ocr_reader = None
_model_lock = threading.Lock()

CAMERA_LOCATIONS = {
    "CAM_001": {
        "location": "Panipat Toll Plaza, NH-44, Panipat, Haryana, India",
        "latitude": 29.421500,
        "longitude": 76.968200,
        "color": (0, 255, 180),        # Cyan-green tint for this camera
        "road_color": (28, 36, 52),
        "bg_color": (10, 14, 24),
        "label": "TOLL PLAZA · NH-44",
    },
    "CAM_004": {
        "location": "GT Road, Bus Stand Chowk, Panipat, Haryana, India",
        "latitude": 29.390900,
        "longitude": 76.963500,
        "color": (0, 220, 255),        # Yellow-cyan
        "road_color": (30, 34, 48),
        "bg_color": (8, 12, 20),
        "label": "GT ROAD · BUS STAND",
    },
    "CAM_005": {
        "location": "Sector 18 Main Gate, Panipat, Haryana, India",
        "latitude": 29.378400,
        "longitude": 76.981200,
        "color": (180, 255, 100),      # Lime green
        "road_color": (24, 32, 44),
        "bg_color": (6, 10, 18),
        "label": "SECTOR-18 MAIN GATE",
    },
    "CAM_002": {
        "location": "Ibrahim Lodi Road, Model Town, Panipat, Haryana, India",
        "latitude": 29.395000,
        "longitude": 76.975000,
        "color": (255, 140, 60),       # Orange tint
        "road_color": (32, 28, 44),
        "bg_color": (12, 8, 20),
        "label": "MODEL TOWN · IBRAHIM LODI RD",
    },
}

DEFAULT_CAMERA_LOC = {
    "location": "GT Road, Panipat, Haryana, India",
    "latitude": 29.390900,
    "longitude": 76.963500,
}

VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck

# Plate pool per camera to simulate different city zones
CAMERA_PLATES = {
    "CAM_001": ["HR06A1245", "DL10CB4921", "HR06W9920", "DL3CAV8834", "HR06BC0012"],
    "CAM_004": ["UP16BE7789", "CH01AB9988", "HR55AK3310", "DL8CAT7721", "HR06A1245"],
    "CAM_005": ["RJ14CD1122", "PB10AQ4590", "HR06BC0012", "MH43AX1982", "CH01AB9988"],
    "CAM_002": ["DL10CB4921", "UP32EE1234", "HR58AZ7776", "DL9CAQ2250", "RJ14CD1122"],
}

# Vehicle colour themes per camera
CAM_VEHICLE_COLORS = {
    "CAM_001": [(60, 90, 140), (100, 60, 60), (50, 110, 80)],
    "CAM_004": [(70, 80, 130), (110, 70, 50), (55, 100, 90)],
    "CAM_005": [(80, 110, 60), (60, 80, 130), (110, 90, 50)],
    "CAM_002": [(130, 60, 80), (70, 120, 100), (90, 70, 120)],
}


def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        with _model_lock:
            if _yolo_model is None:
                try:
                    from ultralytics import YOLO
                    _yolo_model = YOLO("yolov8n.pt")
                    print("[Vision] YOLOv8 loaded successfully.")
                except Exception as e:
                    print(f"[Vision] Error loading YOLO: {e}")
                    _yolo_model = None
    return _yolo_model


def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        with _model_lock:
            if _ocr_reader is None:
                try:
                    import easyocr
                    _ocr_reader = easyocr.Reader(["en"], gpu=False)
                    print("[Vision] EasyOCR loaded successfully.")
                except Exception as e:
                    print(f"[Vision] Error loading EasyOCR: {e}")
                    _ocr_reader = None
    return _ocr_reader


def clean_plate_text(text: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    if 4 <= len(cleaned) <= 11 and any(c.isalpha() for c in cleaned) and any(c.isdigit() for c in cleaned):
        return cleaned
    return ""


def extract_plate_with_confidence(crop_img):
    reader = get_ocr_reader()
    if reader is None or crop_img is None or crop_img.size == 0:
        return None, 0.0

    try:
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)

        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        candidate_crops = []
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                if 1.5 <= aspect_ratio <= 6.5 and w > 35 and h > 10:
                    candidate_crops.append(crop_img[y : y + h, x : x + w])

        vh, vw = crop_img.shape[:2]
        candidate_crops.append(crop_img[int(vh * 0.4) : vh, 0:vw])
        candidate_crops.append(crop_img)

        for candidate in candidate_crops:
            if candidate.size == 0:
                continue
            ocr_results = reader.readtext(candidate, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            valid_blocks = [res for res in ocr_results if res[2] > 0.25]
            if valid_blocks:
                combined_text = "".join([res[1] for res in valid_blocks])
                avg_conf = sum([res[2] for res in valid_blocks]) / len(valid_blocks)
                plate = clean_plate_text(combined_text)
                if plate:
                    return plate, round(float(avg_conf), 3)
    except Exception as e:
        print(f"[Vision] OCR extraction error: {e}")

    return None, 0.0


def calculate_direction(track_id, current_box, previous_boxes):
    x1, y1, x2, y2 = current_box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    area = (x2 - x1) * (y2 - y1)

    if track_id not in previous_boxes:
        previous_boxes[track_id] = (cx, cy, area)
        return "Northbound"

    prev_cx, prev_cy, prev_area = previous_boxes[track_id]
    dx = cx - prev_cx
    dy = cy - prev_cy
    area_ratio = area / max(prev_area, 1.0)
    previous_boxes[track_id] = (cx, cy, area)

    if area_ratio > 1.05 or dy > 6:
        return "Southbound"
    elif area_ratio < 0.95 or dy < -6:
        return "Northbound"
    elif dx > 8:
        return "Eastbound"
    elif dx < -8:
        return "Westbound"
    return "Stationary"


def detect_best_working_camera() -> int:
    """Finds the camera index that produces non-zero frames. Skips probing on headless cloud environments."""
    if os.name != "nt":
        import glob
        video_devs = glob.glob("/dev/video*")
        if not video_devs:
            print("[Camera] Headless cloud server detected (no /dev/video* devices). Running in full City Simulation Mode.")
            return -1

    for idx in [1, 0]:
        try:
            if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
                c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            else:
                c = cv2.VideoCapture(idx)
            if c.isOpened():
                for _ in range(4):
                    ret, f = c.read()
                if ret and f is not None and f.mean() > 15.0:
                    c.release()
                    print(f"[Camera] Detected active physical webcam at index {idx} (mean brightness: {f.mean():.1f})")
                    return idx
                c.release()
        except Exception:
            pass
    return -1 if os.name != "nt" else 1


# ─────────────────────────────────────────────────────────────────────────────
# Per-camera simulation renderer
# ─────────────────────────────────────────────────────────────────────────────

def _draw_city_surveillance_frame(cam_id: str, logged_cooldown: dict):
    """
    Renders a unique city-surveillance-style synthetic frame for each camera node.
    Each camera has its own colour scheme, vehicle mix, and ANPR overlay.
    Returns the rendered frame (numpy array BGR).
    """
    info = CAMERA_LOCATIONS.get(cam_id, DEFAULT_CAMERA_LOC)
    hud_color = info.get("color", (0, 255, 180))
    road_color = info.get("road_color", (28, 36, 52))
    bg_color = info.get("bg_color", (10, 14, 24))
    cam_label = info.get("label", cam_id)
    plates = CAMERA_PLATES.get(cam_id, ["HR06A1245"])
    veh_colors = CAM_VEHICLE_COLORS.get(cam_id, [(60, 90, 140)])

    w, h = 640, 360
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = bg_color

    # ── Sky gradient (top 30%) ──────────────────────────────────────────────
    sky_h = int(h * 0.30)
    for row in range(sky_h):
        frac = row / float(sky_h)
        r = int(bg_color[2] * (1 - frac) + road_color[2] * frac)
        g = int(bg_color[1] * (1 - frac) + road_color[1] * frac)
        b = int(bg_color[0] * (1 - frac) + road_color[0] * frac)
        frame[row, :] = (b, g, r)

    # ── Road surface (perspective trapezoid) ───────────────────────────────
    horizon_y = int(h * 0.34)
    pts_road = np.array([
        [int(w * 0.25), horizon_y],
        [int(w * 0.75), horizon_y],
        [w, h],
        [0, h]
    ], np.int32)
    cv2.fillPoly(frame, [pts_road], road_color)

    # Road edge lines
    cv2.line(frame, (int(w * 0.25), horizon_y), (0, h), (60, 70, 90), 2)
    cv2.line(frame, (int(w * 0.75), horizon_y), (w, h), (60, 70, 90), 2)

    # ── Animated lane dashes ───────────────────────────────────────────────
    t = time.time()
    dash_offset = int((t * 90) % 80)
    cx_road = w // 2
    for y in range(horizon_y + dash_offset, h, 80):
        frac = max(0, (y - horizon_y) / float(h - horizon_y))
        dw = int(3 + 7 * frac)
        dh = int(10 + 25 * frac)
        cv2.rectangle(frame, (cx_road - dw, y), (cx_road + dw, y + dh), (180, 180, 180), -1)

    # ── Buildings silhouette in background ────────────────────────────────
    rng = np.random.RandomState(abs(hash(cam_id)) % (2**32))
    num_buildings = 8
    for i in range(num_buildings):
        bx = int(i * (w / num_buildings))
        bw_b = int(w / num_buildings) - 4
        bh_b = rng.randint(20, 80)
        by = horizon_y - bh_b
        bld_col = (
            int(bg_color[0] * 1.5),
            int(bg_color[1] * 1.5),
            int(bg_color[2] * 1.5),
        )
        cv2.rectangle(frame, (bx, by), (bx + bw_b, horizon_y), bld_col, -1)
        # Windows
        for wy in range(by + 4, horizon_y - 4, 10):
            for wx in range(bx + 4, bx + bw_b - 4, 10):
                win_on = rng.rand() > 0.4
                if win_on:
                    cv2.rectangle(frame, (wx, wy), (wx + 5, wy + 5), (200, 180, 100), -1)

    # ── Multi-lane vehicle simulation ─────────────────────────────────────
    # 2 lanes: left and right of center
    lane_offsets = [-0.18, 0.18]   # fraction of road width from center
    for lane_i, lane_off in enumerate(lane_offsets):
        # Each lane has a different phase so they don't overlap
        phase = lane_i * 0.45
        cycle_pos = ((t * 0.30 + phase) % 1.0)
        car_y = int(horizon_y + 20 + cycle_pos * (h - horizon_y - 50))
        frac = max(0.01, (car_y - horizon_y) / float(h - horizon_y))
        car_w = int(50 + frac * 120)
        car_h = int(30 + frac * 80)
        car_cx = int(cx_road + lane_off * w + math.sin(t * 0.8 + lane_i) * 12 * frac)
        car_x = car_cx - car_w // 2

        # Pick vehicle colour based on lane + time cycle
        vcol_idx = (lane_i + int(t // 12)) % len(veh_colors)
        veh_col = veh_colors[vcol_idx]

        # Body
        cv2.rectangle(frame, (car_x, car_y), (car_x + car_w, car_y + car_h), veh_col, -1)
        # Roof
        roof_x1 = car_x + int(car_w * 0.15)
        roof_x2 = car_x + int(car_w * 0.85)
        roof_y1 = car_y + int(car_h * 0.1)
        roof_y2 = car_y + int(car_h * 0.45)
        roof_col = tuple(max(0, c - 20) for c in veh_col)
        cv2.rectangle(frame, (roof_x1, roof_y1), (roof_x2, roof_y2), roof_col, -1)
        # Windshield
        cv2.rectangle(frame,
            (car_x + int(car_w*0.18), car_y + int(car_h*0.12)),
            (car_x + int(car_w*0.82), car_y + int(car_h*0.43)),
            (70, 100, 130), -1)
        # Headlights (bottom)
        hl_y = car_y + car_h - int(car_h * 0.15)
        cv2.circle(frame, (car_x + int(car_w * 0.18), hl_y), max(2, int(4 * frac)), (240, 255, 255), -1)
        cv2.circle(frame, (car_x + car_w - int(car_w * 0.18), hl_y), max(2, int(4 * frac)), (240, 255, 255), -1)

        # ── License plate ────────────────────────────────────────────────
        plate_w = max(30, int(car_w * 0.50))
        plate_h = max(10, int(plate_w * 0.28))
        plate_x = car_x + (car_w - plate_w) // 2
        plate_y = car_y + int(car_h * 0.62)
        cv2.rectangle(frame, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), (245, 245, 245), -1)
        cv2.rectangle(frame, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), (0, 0, 0), 1)

        plate_idx = (lane_i + int(t // 9)) % len(plates)
        demo_plate = plates[plate_idx]

        font_scale = max(0.25, 0.40 * frac)
        cv2.putText(frame, demo_plate,
            (plate_x + 3, plate_y + plate_h - 3),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), max(1, int(frac * 2)))

        # ── Detection bounding box ────────────────────────────────────────
        box_pad = 4
        cv2.rectangle(frame,
            (car_x - box_pad, car_y - box_pad),
            (car_x + car_w + box_pad, car_y + car_h + box_pad),
            hud_color, 1)
        # Label above box
        lbl = f"{demo_plate}  [{0.91 + 0.05 * frac:.2f}]"
        lbl_x = car_x - box_pad
        lbl_y = car_y - box_pad - 6
        if lbl_y > 12 and frac > 0.15:
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.rectangle(frame, (lbl_x, lbl_y - th - 2), (lbl_x + tw + 4, lbl_y + 2), (0, 0, 0), -1)
            cv2.putText(frame, lbl, (lbl_x + 2, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, hud_color, 1)

        # ── Log to database (throttled) ───────────────────────────────────
        now_sec = time.time()
        log_key = f"{cam_id}_{demo_plate}"
        if log_key not in logged_cooldown or (now_sec - logged_cooldown[log_key]) > 20:
            now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
            db_manager.add_detection(
                cam_id, demo_plate,
                location=info["location"],
                latitude=info["latitude"],
                longitude=info["longitude"],
                direction="Southbound",
                timestamp=now_ist,
                confidence=round(0.91 + 0.05 * frac, 3),
                status="high_confidence",
            )
            logged_cooldown[log_key] = now_sec

    # ── Cyber scan line overlay ────────────────────────────────────────────
    scan_y = int((t * 60) % h)
    cv2.line(frame, (0, scan_y), (w, scan_y), (*hud_color[:2], 60), 1)

    # ── Grid lines ────────────────────────────────────────────────────────
    for gy in range(0, h, 50):
        cv2.line(frame, (0, gy), (w, gy), (18, 24, 36), 1)

    # ── Top HUD bar ───────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 0), -1)
    hud_top = f"● LIVE  {cam_id}  |  {cam_label}"
    cv2.putText(frame, hud_top, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.44, hud_color, 1)
    # Timestamp top-right
    ts = datetime.now(IST).strftime("%H:%M:%S IST")
    (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.putText(frame, ts, (w - tw - 8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 200, 220), 1)
    # Recording dot blink
    if int(t * 2) % 2 == 0:
        cv2.circle(frame, (w - tw - 20, 14), 4, (0, 0, 255), -1)

    # ── Bottom HUD bar ────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, h - 24), (w, h), (0, 0, 0), -1)
    loc_text = f"GPS: {info['latitude']:.4f}, {info['longitude']:.4f}  |  {info['location'][:42]}"
    cv2.putText(frame, loc_text, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 180, 200), 1)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Per-camera stream generator (singleton per cam_id)
# ─────────────────────────────────────────────────────────────────────────────

class CameraStreamNode:
    """Independent MJPEG generator for a single camera node."""

    def __init__(self, cam_id: str):
        self.cam_id = cam_id
        self.is_running = True
        self.mode = "simulation"
        self.logged_cooldown: dict = {}

    def generate(self):
        while self.is_running:
            if self.mode == "webcam" and stream_manager.mode == "webcam":
                frame = stream_manager.get_processed_frame()
            else:
                frame = _draw_city_surveillance_frame(self.cam_id, self.logged_cooldown)
            ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not ret:
                time.sleep(0.04)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            time.sleep(0.04)   # ~25 FPS


# Singleton nodes — one per camera
_cam_nodes: dict[str, CameraStreamNode] = {}
_cam_nodes_lock = threading.Lock()


def get_camera_node(cam_id: str) -> CameraStreamNode:
    with _cam_nodes_lock:
        if cam_id not in _cam_nodes:
            _cam_nodes[cam_id] = CameraStreamNode(cam_id)
        return _cam_nodes[cam_id]


# ─────────────────────────────────────────────────────────────────────────────
# Legacy VideoStreamManager (kept for /api/video_feed + webcam support)
# ─────────────────────────────────────────────────────────────────────────────

class VideoStreamManager:
    """Manages the primary (selected) camera stream with optional webcam mode."""

    def __init__(self):
        self.active_camera_id = "CAM_004"
        self.camera_index = detect_best_working_camera()
        self.cap = None
        self.is_running = True
        self.logged_cooldown = {}
        self.previous_boxes = {}
        self.tracked_vehicles = {}
        self.frame_count = 0
        self.mode = "simulation"
        self._init_camera()

    def _init_camera(self):
        if self.camera_index is None or self.camera_index < 0:
            self.cap = None
            return

        try:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
            if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                for _ in range(5):
                    self.cap.read()
                print(f"[Camera] Initialized webcam hardware index {self.camera_index}")
        except Exception as e:
            print(f"[Camera] Error initializing hardware webcam: {e}")
            self.cap = None

    def set_mode(self, mode: str):
        self.mode = mode
        if self.mode == "webcam":
            if self.cap is None or not self.cap.isOpened():
                self._init_camera()
        print(f"[Camera] Feed mode switched to: {self.mode}")

    def set_camera(self, cam_id: str, camera_index: int = None):
        self.active_camera_id = cam_id
        if camera_index is not None:
            self.camera_index = camera_index
        else:
            self.camera_index = detect_best_working_camera()
        self._init_camera()

    def get_processed_frame(self):
        self.frame_count += 1

        if self.mode == "simulation":
            return _draw_city_surveillance_frame(self.active_camera_id, self.logged_cooldown)

        cam_info = CAMERA_LOCATIONS.get(self.active_camera_id, DEFAULT_CAMERA_LOC)

        if self.cap is None or not self.cap.isOpened():
            self._init_camera()

        if self.cap is None or not self.cap.isOpened():
            return _draw_city_surveillance_frame(self.active_camera_id, self.logged_cooldown)

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return _draw_city_surveillance_frame(self.active_camera_id, self.logged_cooldown)

        mean_brightness = float(np.mean(frame))
        if mean_brightness < 10.0:
            if self.camera_index == 0:
                self.camera_index = 1
                self._init_camera()
                if self.cap and self.cap.isOpened():
                    ret2, frame2 = self.cap.read()
                    if ret2 and frame2 is not None and float(np.mean(frame2)) > 15.0:
                        frame = frame2
                    else:
                        return _draw_city_surveillance_frame(self.active_camera_id, self.logged_cooldown)
                else:
                    return _draw_city_surveillance_frame(self.active_camera_id, self.logged_cooldown)
            else:
                return _draw_city_surveillance_frame(self.active_camera_id, self.logged_cooldown)

        model = get_yolo_model()
        if model is not None:
            try:
                results = model.track(frame, classes=VEHICLE_CLASSES, persist=True, verbose=False)
                boxes = results[0].boxes if (results and results[0].boxes is not None) else []

                if len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        track_id = int(box.id[0]) if box.id is not None else 0
                        yolo_conf = float(box.conf[0]) if box.conf is not None else 0.0

                        direction = calculate_direction(track_id, (x1, y1, x2, y2), self.previous_boxes)

                        if track_id not in self.tracked_vehicles or self.frame_count % 6 == 0:
                            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                            if crop.size > 0:
                                plate, ocr_conf = extract_plate_with_confidence(crop)
                                if plate:
                                    final_conf = round((yolo_conf + ocr_conf) / 2, 3)
                                    status = "high_confidence" if final_conf >= 0.60 else "low_confidence"
                                    now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
                                    self.tracked_vehicles[track_id] = {
                                        "plate": plate, "confidence": final_conf,
                                        "status": status, "direction": direction,
                                    }
                                    now_sec = time.time()
                                    if plate not in self.logged_cooldown or (now_sec - self.logged_cooldown[plate]) > 15:
                                        db_manager.add_detection(
                                            self.active_camera_id, plate,
                                            location=cam_info["location"],
                                            latitude=cam_info["latitude"],
                                            longitude=cam_info["longitude"],
                                            direction=direction, timestamp=now_ist,
                                            confidence=final_conf, status=status,
                                        )
                                        self.logged_cooldown[plate] = now_sec

                        if track_id in self.tracked_vehicles:
                            info = self.tracked_vehicles[track_id]
                            label = f"ID:{track_id} {info['plate']} ({direction}) [{info['confidence']}]"
                            color = (0, 255, 0)
                        else:
                            label = f"ID:{track_id} Scanning... ({direction})"
                            color = (0, 165, 255)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                elif self.frame_count % 15 == 0:
                    plate, ocr_conf = extract_plate_with_confidence(frame)
                    if plate:
                        status = "high_confidence" if ocr_conf >= 0.60 else "low_confidence"
                        now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
                        now_sec = time.time()
                        if plate not in self.logged_cooldown or (now_sec - self.logged_cooldown[plate]) > 15:
                            db_manager.add_detection(
                                self.active_camera_id, plate,
                                location=cam_info["location"], latitude=cam_info["latitude"],
                                longitude=cam_info["longitude"], direction="Close-Up Scan",
                                timestamp=now_ist, confidence=ocr_conf, status=status,
                            )
                            self.logged_cooldown[plate] = now_sec
                        cv2.putText(frame, f"Plate: {plate} ({ocr_conf})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            except Exception as e:
                print(f"[Vision] Frame processing error: {e}")

        cv2.putText(frame, f"LIVE WEBCAM [{self.camera_index}] | {self.active_camera_id}",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        footer = f"{self.active_camera_id} | {cam_info['location'][:40]}"
        cv2.putText(frame, footer, (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        return frame

    def generate_mjpeg_stream(self):
        while self.is_running:
            frame = self.get_processed_frame()
            ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                time.sleep(0.04)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            time.sleep(0.04)


# Singleton instance (legacy primary stream)
stream_manager = VideoStreamManager()


def scan_single_image(image_bytes: bytes, cam_id: str = "CAM_004"):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Invalid image file format"}

    model = get_yolo_model()
    detected_plates = []
    cam_info = CAMERA_LOCATIONS.get(cam_id, DEFAULT_CAMERA_LOC)
    now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

    if model is not None:
        results = model(img, classes=VEHICLE_CLASSES, verbose=False)
        boxes = results[0].boxes if (results and results[0].boxes is not None) else []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            yolo_conf = float(box.conf[0]) if box.conf is not None else 0.0
            crop = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
            if crop.size > 0:
                plate, ocr_conf = extract_plate_with_confidence(crop)
                if plate:
                    final_conf = round((yolo_conf + ocr_conf) / 2, 3)
                    status = "high_confidence" if final_conf >= 0.60 else "low_confidence"
                    detected_plates.append({"plate": plate, "confidence": final_conf, "status": status, "box": [x1, y1, x2, y2]})
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(img, f"{plate} [{final_conf}]", (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if not detected_plates:
        plate, ocr_conf = extract_plate_with_confidence(img)
        if plate:
            status = "high_confidence" if ocr_conf >= 0.60 else "low_confidence"
            detected_plates.append({"plate": plate, "confidence": ocr_conf, "status": status, "box": [10, 10, img.shape[1]-10, img.shape[0]-10]})
            cv2.putText(img, f"Plate: {plate} [{ocr_conf}]", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    for item in detected_plates:
        db_manager.add_detection(
            cam_id, item["plate"],
            location=cam_info["location"], latitude=cam_info["latitude"],
            longitude=cam_info["longitude"], direction="Direct Scan",
            timestamp=now_ist, confidence=item["confidence"], status=item["status"],
        )

    _, buffer = cv2.imencode(".jpg", img)
    b64_image = base64.b64encode(buffer).decode("utf-8")

    return {
        "success": True,
        "plates_detected": detected_plates,
        "total_detected": len(detected_plates),
        "annotated_image": f"data:image/jpeg;base64,{b64_image}",
        "timestamp": now_ist
    }
