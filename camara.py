from datetime import datetime
import re
import time
import urllib.request
import json

# Import your MongoDB add_detection function and IST timezone from car.py
from car import IST, add_detection
import cv2
import easyocr
import geocoder
from geopy.geocoders import Nominatim
from ultralytics import YOLO

# 1. Initialize Models (COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck)
VEHICLE_CLASSES = [2, 3, 5, 7]
model = YOLO("yolov8n.pt")
reader = easyocr.Reader(["en"], gpu=False)  # Set gpu=True if using a CUDA GPU
geolocator = Nominatim(user_agent="anpr_live_camera_v2")

# 2. Exact Camera GPS Registry (Add your fixed camera locations here for 100% accuracy)
CAMERA_LOCATIONS = {
    "CAM_001": {
        "location": "Panipat Toll Plaza, NH-44, Panipat, Haryana, India",
        "latitude": 29.421500,
        "longitude": 76.968200,
    },
    "CAM_004": {
        "location": "GT Road, Bus Stand Chowk, Panipat, Haryana, India",
        "latitude": 29.390900,
        "longitude": 76.963500,
    },
    "CAM_005": {
        "location": "Sector 18 Main Gate, Panipat, Haryana, India",
        "latitude": 29.378400,
        "longitude": 76.981200,
    },
}

DEFAULT_CAMERA_LOC = {
    "location": "GT Road, Panipat, Haryana, India",
    "latitude": 29.390900,
    "longitude": 76.963500,
}


def _reverse_geocode_address(lat: float, lon: float, fallback_address: str) -> str:
    """Converts exact latitude & longitude into a full street/landmark address."""
    try:
        loc = geolocator.reverse(f"{lat}, {lon}", exactly_one=True, language="en", timeout=5)
        if loc and loc.address:
            return loc.address
    except Exception:
        pass
    return fallback_address


def get_camera_location(cam_id: str = None, custom_lat: float = None, custom_lon: float = None) -> dict:
    """Resolves accurate location, latitude, and longitude using:
    1. Custom coordinates if passed
    2. Pre-configured CAMERA_LOCATIONS for fixed cameras (CAM_004, CAM_005, etc.)
    3. Multi-provider live geolocation (ipwho.is -> ip-api.com -> geocoder) + Reverse Geocoding
    """
    # Tier 1: Explicit GPS coordinates passed by user
    if custom_lat is not None and custom_lon is not None:
        lat, lon = round(float(custom_lat), 6), round(float(custom_lon), 6)
        address = _reverse_geocode_address(lat, lon, DEFAULT_CAMERA_LOC["location"])
        return {"location": address, "latitude": lat, "longitude": lon}

    # Tier 2: Check if cam_id has fixed GPS coordinates in CAMERA_LOCATIONS
    if cam_id and cam_id in CAMERA_LOCATIONS:
         preset = CAMERA_LOCATIONS[cam_id].copy()
         preset["latitude"] = round(float(preset["latitude"]), 6)
         preset["longitude"] = round(float(preset["longitude"]), 6)
         return preset

    # Tier 3A: Live geolocation via ipwho.is (high reliability, no API key needed)
    try:
        req = urllib.request.Request("https://ipwho.is/", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("success") and data.get("latitude") and data.get("longitude"):
                lat = round(float(data["latitude"]), 6)
                lon = round(float(data["longitude"]), 6)
                fallback = f"{data.get('city', '')}, {data.get('region', '')}, {data.get('country', '')}".strip(", ")
                address = _reverse_geocode_address(lat, lon, fallback)
                return {"location": address, "latitude": lat, "longitude": lon}
    except Exception:
        pass

    # Tier 3B: Live geolocation via ip-api.com
    try:
        req = urllib.request.Request("http://ip-api.com/json/", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("status") == "success" and data.get("lat") and data.get("lon"):
                lat = round(float(data["lat"]), 6)
                lon = round(float(data["lon"]), 6)
                fallback = f"{data.get('city', '')}, {data.get('regionName', '')}, {data.get('country', '')}".strip(", ")
                address = _reverse_geocode_address(lat, lon, fallback)
                return {"location": address, "latitude": lat, "longitude": lon}
    except Exception:
        pass

    # Tier 3C: Fallback to geocoder.ip("me")
    try:
        g = geocoder.ip("me")
        if g.latlng:
            lat, lon = round(float(g.latlng[0]), 6), round(float(g.latlng[1]), 6)
            fallback = f"{g.city}, {g.state}, {g.country}"
            address = _reverse_geocode_address(lat, lon, fallback)
            return {"location": address, "latitude": lat, "longitude": lon}
    except Exception:
        pass

    return DEFAULT_CAMERA_LOC.copy()


def clean_plate_text(text: str) -> str:
    """Keeps only alphanumeric characters and verifies valid plate length."""
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    if (
        4 <= len(cleaned) <= 11
        and any(c.isalpha() for c in cleaned)
        and any(c.isdigit() for c in cleaned)
    ):
        return cleaned
    return ""


def extract_plate_with_confidence(crop_img):
    """Finds plate contours in the crop and returns (plate_number, confidence)."""
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    contours, _ = cv2.findContours(
        edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    candidate_crops = []
    for contour in contours:
        approx = cv2.approxPolyDP(
            contour, 0.02 * cv2.arcLength(contour, True), True
        )
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if 1.5 <= aspect_ratio <= 6.5 and w > 35 and h > 10:
                candidate_crops.append(crop_img[y : y + h, x : x + w])

    # Fallback: scan lower 60% of the vehicle and the full crop
    vh, vw = crop_img.shape[:2]
    candidate_crops.append(crop_img[int(vh * 0.4) : vh, 0:vw])
    candidate_crops.append(crop_img)

    for candidate in candidate_crops:
        if candidate.size == 0:
            continue
        ocr_results = reader.readtext(
            candidate, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        valid_blocks = [res for res in ocr_results if res[2] > 0.25]
        if valid_blocks:
            combined_text = "".join([res[1] for res in valid_blocks])
            avg_conf = sum([res[2] for res in valid_blocks]) / len(valid_blocks)
            plate = clean_plate_text(combined_text)
            if plate:
                return plate, round(float(avg_conf), 3)

    return None, 0.0


def calculate_direction(track_id, current_box, previous_boxes):
    """Determines travel direction from bounding box movement across frames."""
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


def get_status(confidence: float) -> str:
    return "high_confidence" if confidence >= 0.60 else "low_confidence"


def run_live_camera(
    camera_index=0,
    cam_id="CAM_004",
    custom_lat: float = None,
    custom_lon: float = None,
):
    """Streams from the camera, detects single/multiple vehicles, and logs to MongoDB via add_detection."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {camera_index}")

    # Fetch accurate location, latitude, and longitude for this camera
    cam_loc = get_camera_location(
        cam_id=cam_id, custom_lat=custom_lat, custom_lon=custom_lon
    )
    location_name = str(cam_loc["location"])
    latitude_val = float(cam_loc["latitude"])
    longitude_val = float(cam_loc["longitude"])

    previous_boxes = {}
    tracked_vehicles = {}  # track_id -> latest plate info for screen display
    logged_cooldown = {}   # plate -> last timestamp saved (prevents duplicate spam)
    frame_count = 0

    print(f"[{cam_id}] Live Camera Started")
    print(f"Location  : {location_name}")
    print(f"Latitude  : {latitude_val}")
    print(f"Longitude : {longitude_val}")
    print("Press 'q' in the video window to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Track single or multiple vehicles in the live frame
        results = model.track(
            frame, classes=VEHICLE_CLASSES, persist=True, verbose=False
        )
        boxes = (
            results[0].boxes
            if (results and results[0].boxes is not None)
            else []
        )

        # CASE 1: One or multiple vehicles detected in the camera frame
        if len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else 0
                yolo_conf = float(box.conf[0]) if box.conf is not None else 0.0

                direction = calculate_direction(
                    track_id, (x1, y1, x2, y2), previous_boxes
                )

                # Run OCR every 5th frame or if this vehicle doesn't have a plate yet
                if track_id not in tracked_vehicles or frame_count % 5 == 0:
                    crop = frame[
                        max(0, y1) : min(frame.shape[0], y2),
                        max(0, x1) : min(frame.shape[1], x2),
                    ]
                    if crop.size > 0:
                        plate, ocr_conf = extract_plate_with_confidence(crop)
                        if plate:
                            final_conf = round((yolo_conf + ocr_conf) / 2, 3)
                            status = get_status(final_conf)
                            now_ist = datetime.now(IST).strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )

                            tracked_vehicles[track_id] = {
                                "plate": plate,
                                "confidence": final_conf,
                                "status": status,
                                "direction": direction,
                            }

                            # Save to MongoDB via add_detection from car.py (15s cooldown per unique plate)
                            now_sec = time.time()
                            if (
                                plate not in logged_cooldown
                                or (now_sec - logged_cooldown[plate]) > 15
                            ):
                                add_detection(
                                    cam_id,
                                    plate,
                                    location=location_name,
                                    latitude=latitude_val,
                                    longitude=longitude_val,
                                    direction=direction,
                                    timestamp=now_ist,
                                    confidence=final_conf,
                                    status=status,
                                )
                                logged_cooldown[plate] = now_sec
                                print(
                                    f"[SAVED TO MONGODB] Cam: {cam_id} | Plate: {plate} | "
                                    f"Loc: {location_name} ({latitude_val}, {longitude_val}) | "
                                    f"Dir: {direction} | Conf: {final_conf} | Time: {now_ist}"
                                )

                # Draw bounding box and telemetry on the live feed
                if track_id in tracked_vehicles:
                    info = tracked_vehicles[track_id]
                    label = f"ID:{track_id} {info['plate']} ({direction}) [{info['confidence']}]"
                    color = (0, 255, 0)
                else:
                    label = f"ID:{track_id} Scanning... ({direction})"
                    color = (0, 165, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

        # CASE 2: Close-up plate held directly in front of the webcam (no full car body visible)
        elif frame_count % 15 == 0:
            plate, ocr_conf = extract_plate_with_confidence(frame)
            if plate:
                status = get_status(ocr_conf)
                now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
                now_sec = time.time()

                if (
                    plate not in logged_cooldown
                    or (now_sec - logged_cooldown[plate]) > 15
                ):
                    add_detection(
                        cam_id,
                        plate,
                        location=location_name,
                        latitude=latitude_val,
                        longitude=longitude_val,
                        direction="Close-Up / Front",
                        timestamp=now_ist,
                        confidence=ocr_conf,
                        status=status,
                    )
                    logged_cooldown[plate] = now_sec
                    print(
                        f"[SAVED TO MONGODB (Close-Up)] Cam: {cam_id} | Plate: {plate} | "
                        f"Loc: {location_name} ({latitude_val}, {longitude_val}) | "
                        f"Conf: {ocr_conf} | Time: {now_ist}"
                    )

                cv2.putText(
                    frame,
                    f"Plate: {plate} ({ocr_conf})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

        # Draw live GPS & Location footer bar on the camera window
        footer_text = f"{cam_id} | {location_name[:45]} | Lat: {latitude_val}, Lon: {longitude_val}"
        cv2.putText(
            frame,
            footer_text,
            (10, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )

        cv2.imshow(f"Live ANPR Camera - {cam_id}", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Pass camera_index=0 for default webcam and cam_id (e.g., "CAM_004" or "CAM_005")
    run_live_camera(camera_index=0, cam_id="CAM_004")
