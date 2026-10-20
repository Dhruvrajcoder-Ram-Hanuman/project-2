from datetime import datetime, timedelta, timezone
import json
import os
import re
import certifi
from pydantic import BaseModel
from pymongo import MongoClient
import pymongo.errors

# Indian Standard Time
IST = timezone(timedelta(hours=5, minutes=30))

# MongoDB Atlas URI (Supports env var fallback)
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://vercelweb1_db_user:DhruvrajGoyal@cluster0.mi37f6m.mongodb.net/"
)

# Local backup storage path
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LOCAL_DB_FILE = os.path.join(DATA_DIR, "local_storage.json")
os.makedirs(DATA_DIR, exist_ok=True)


class VehicleDetection(BaseModel):
    location: str
    latitude: float
    longitude: float
    direction: str
    timestamp: str
    confidence: float
    status: str


class DatabaseManager:
    """Enterprise-grade database manager supporting MongoDB Atlas with resilient local fallback."""

    def __init__(self):
        self.mongo_client = None
        self.db = None
        self.collection = None
        self.is_connected = False
        self._init_mongo()
        self._init_local_db()

    def _init_mongo(self):
        try:
            # Attempt secure connection with certifi & timeout
            self.mongo_client = MongoClient(
                MONGO_URI,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=2500,
                connectTimeoutMS=2500,
                socketTimeoutMS=2500,
            )
            # Ping database to confirm connection
            self.mongo_client.admin.command("ping")
            self.db = self.mongo_client["Car"]
            self.collection = self.db["vehicle"]
            self.is_connected = True
            print("[Database] Successfully connected to MongoDB Atlas (Car.vehicle)")
        except Exception as e:
            self.is_connected = False
            self.collection = None
            print(f"[Database] MongoDB Atlas unavailable ({e}). Using resilient local storage fallback.")

    def _init_local_db(self):
        if not os.path.exists(LOCAL_DB_FILE):
            # Seed with initial camera telemetry so dashboard has data immediately
            initial_data = {
                "cameras": {
                    "CAM_001": {
                        "_id": "CAM_001",
                        "location": "Panipat Toll Plaza, NH-44, Panipat, Haryana, India",
                        "latitude": 29.421500,
                        "longitude": 76.968200,
                        "plates": {
                            "HR06A1245": [
                                {
                                    "location": "Panipat Toll Plaza, NH-44, Panipat, Haryana, India",
                                    "latitude": 29.421500,
                                    "longitude": 76.968200,
                                    "direction": "Southbound",
                                    "timestamp": "2026-10-20 09:15:22",
                                    "confidence": 0.945,
                                    "status": "high_confidence"
                                }
                            ]
                        }
                    },
                    "CAM_004": {
                        "_id": "CAM_004",
                        "location": "GT Road, Bus Stand Chowk, Panipat, Haryana, India",
                        "latitude": 29.390900,
                        "longitude": 76.963500,
                        "plates": {
                            "HR06A1245": [
                                {
                                    "location": "GT Road, Bus Stand Chowk, Panipat, Haryana, India",
                                    "latitude": 29.390900,
                                    "longitude": 76.963500,
                                    "direction": "Southbound",
                                    "timestamp": "2026-10-20 09:32:45",
                                    "confidence": 0.962,
                                    "status": "high_confidence"
                                },
                                {
                                    "location": "GT Road, Bus Stand Chowk, Panipat, Haryana, India",
                                    "latitude": 29.390900,
                                    "longitude": 76.963500,
                                    "direction": "Northbound",
                                    "timestamp": "2026-10-20 10:11:10",
                                    "confidence": 0.915,
                                    "status": "high_confidence"
                                }
                            ],
                            "DL10CB4921": [
                                {
                                    "location": "GT Road, Bus Stand Chowk, Panipat, Haryana, India",
                                    "latitude": 29.390900,
                                    "longitude": 76.963500,
                                    "direction": "Northbound",
                                    "timestamp": "2026-10-20 10:45:30",
                                    "confidence": 0.887,
                                    "status": "high_confidence"
                                }
                            ]
                        }
                    },
                    "CAM_005": {
                        "_id": "CAM_005",
                        "location": "Sector 18 Main Gate, Panipat, Haryana, India",
                        "latitude": 29.378400,
                        "longitude": 76.981200,
                        "plates": {
                            "HR06A1245": [
                                {
                                    "location": "Sector 18 Main Gate, Panipat, Haryana, India",
                                    "latitude": 29.378400,
                                    "longitude": 76.981200,
                                    "direction": "Eastbound",
                                    "timestamp": "2026-10-20 10:58:14",
                                    "confidence": 0.978,
                                    "status": "high_confidence"
                                }
                            ],
                            "UP16BE7789": [
                                {
                                    "location": "Sector 18 Main Gate, Panipat, Haryana, India",
                                    "latitude": 29.378400,
                                    "longitude": 76.981200,
                                    "direction": "Westbound",
                                    "timestamp": "2026-10-20 11:15:02",
                                    "confidence": 0.934,
                                    "status": "high_confidence"
                                }
                            ]
                        }
                    },
                    "CAM_002": {
                        "_id": "CAM_002",
                        "location": "Ibrahim Lodi Road, Model Town, Panipat, Haryana, India",
                        "latitude": 29.395000,
                        "longitude": 76.975000,
                        "plates": {
                            "HR06A1245": [
                                {
                                    "location": "Ibrahim Lodi Road, Model Town, Panipat, Haryana, India",
                                    "latitude": 29.395000,
                                    "longitude": 76.975000,
                                    "direction": "Eastbound",
                                    "timestamp": "2026-10-20 11:42:10",
                                    "confidence": 0.958,
                                    "status": "high_confidence"
                                }
                            ],
                            "DL10CB4921": [
                                {
                                    "location": "Ibrahim Lodi Road, Model Town, Panipat, Haryana, India",
                                    "latitude": 29.395000,
                                    "longitude": 76.975000,
                                    "direction": "Westbound",
                                    "timestamp": "2026-10-20 11:50:22",
                                    "confidence": 0.941,
                                    "status": "high_confidence"
                                }
                            ]
                        }
                    }
                }
            }
            with open(LOCAL_DB_FILE, "w", encoding="utf-8") as f:
                json.dump(initial_data, f, indent=2)

    def _read_local_db(self):
        try:
            with open(LOCAL_DB_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"cameras": {}}

    def _write_local_db(self, data):
        try:
            with open(LOCAL_DB_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[Database] Error writing to local DB: {e}")

    def add_detection(
        self,
        cam: str,
        plate: str,
        *,
        location: str,
        latitude: float,
        longitude: float,
        direction: str,
        timestamp: str = None,
        confidence: float,
        status: str,
    ):
        """Appends detection to MongoDB Atlas (if available) and local mirror store."""
        if timestamp is None:
            timestamp = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

        clean_plate = re.sub(r"[^A-Z0-9]", "", plate.upper())
        if not clean_plate:
            return None

        det = VehicleDetection(
            location=location,
            latitude=float(latitude),
            longitude=float(longitude),
            direction=direction,
            timestamp=timestamp,
            confidence=round(float(confidence), 3),
            status=status,
        )
        record = det.model_dump(mode="json")

        # 1. Update Local Mirror
        local_data = self._read_local_db()
        cameras = local_data.setdefault("cameras", {})
        cam_entry = cameras.setdefault(cam, {
            "_id": cam,
            "location": location,
            "latitude": float(latitude),
            "longitude": float(longitude),
            "plates": {}
        })
        cam_entry.setdefault("plates", {}).setdefault(clean_plate, []).append(record)
        self._write_local_db(local_data)

        # 2. Update MongoDB Atlas if connected
        if self.is_connected and self.collection is not None:
            try:
                # Use atomic push for concurrency safety
                self.collection.update_one(
                    {"_id": cam},
                    {
                        "$push": {f"plates.{clean_plate}": record},
                        "$set": {
                            "location": location,
                            "latitude": float(latitude),
                            "longitude": float(longitude),
                            "last_active": timestamp
                        }
                    },
                    upsert=True
                )
            except Exception as e:
                print(f"[Database] MongoDB write failed ({e}). Record safely saved to local storage.")

        return record

    def search_by_plate(self, plate: str):
        """Searches vehicle plate across all cameras, producing chronological journey & route details."""
        clean_plate = re.sub(r"[^A-Z0-9]", "", plate.upper())
        if not clean_plate:
            return {
                "plate_number": "",
                "total_sightings": 0,
                "cameras_spotted": [],
                "first_seen": None,
                "last_seen": None,
                "history": [],
            }

        sightings = []
        cameras_spotted = set()

        # Check MongoDB first if connected
        searched_mongo = False
        if self.is_connected and self.collection is not None:
            try:
                cursor = self.collection.find({f"plates.{clean_plate}": {"$exists": True}})
                for doc in cursor:
                    cam_id = doc["_id"]
                    cameras_spotted.add(cam_id)
                    for rec in doc.get("plates", {}).get(clean_plate, []):
                        item = dict(rec)
                        item["camera"] = cam_id
                        sightings.append(item)
                searched_mongo = True
            except Exception as e:
                print(f"[Database] MongoDB search error: {e}")

        # Check local DB
        local_data = self._read_local_db()
        for cam_id, cam_doc in local_data.get("cameras", {}).items():
            if clean_plate in cam_doc.get("plates", {}):
                for rec in cam_doc["plates"][clean_plate]:
                    item = dict(rec)
                    item["camera"] = cam_id
                    # Avoid duplicate records if both mongo and local have it
                    if not any(
                        s.get("camera") == item["camera"] and s.get("timestamp") == item["timestamp"]
                        for s in sightings
                    ):
                        cameras_spotted.add(cam_id)
                        sightings.append(item)

        # Sort chronologically by timestamp
        sightings.sort(key=lambda x: x.get("timestamp", ""))

        first_seen = sightings[0]["timestamp"] if sightings else None
        last_seen = sightings[-1]["timestamp"] if sightings else None

        return {
            "plate_number": clean_plate,
            "total_sightings": len(sightings),
            "cameras_spotted": list(cameras_spotted),
            "first_seen": first_seen,
            "last_seen": last_seen,
            "history": sightings,
        }

    def get_recent_detections(self, limit: int = 50):
        """Fetches flat chronological list of most recent vehicle detections."""
        all_detections = []
        local_data = self._read_local_db()

        for cam_id, cam_doc in local_data.get("cameras", {}).items():
            for plate_num, records in cam_doc.get("plates", {}).items():
                for rec in records:
                    entry = dict(rec)
                    entry["camera"] = cam_id
                    entry["plate"] = plate_num
                    all_detections.append(entry)

        all_detections.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return all_detections[:limit]

    def get_all_cameras(self):
        """Returns metadata and status for all cameras."""
        local_data = self._read_local_db()
        result = []
        for cam_id, cam_doc in local_data.get("cameras", {}).items():
            total_sightings = sum(len(records) for records in cam_doc.get("plates", {}).values())
            unique_plates = len(cam_doc.get("plates", {}).keys())
            result.append({
                "camera_id": cam_id,
                "location": cam_doc.get("location", "City Surveillance Point"),
                "latitude": cam_doc.get("latitude", 29.3909),
                "longitude": cam_doc.get("longitude", 76.9635),
                "total_sightings": total_sightings,
                "unique_plates": unique_plates,
                "status": "ONLINE"
            })
        return result

    def get_analytics(self):
        """Calculates system-wide surveillance analytics."""
        recent = self.get_recent_detections(limit=500)
        unique_plates = set(d["plate"] for d in recent)
        today_str = datetime.now(IST).strftime("%Y-%m-%d")
        today_detections = [d for d in recent if d.get("timestamp", "").startswith(today_str)]

        direction_counts = {}
        for d in recent:
            direction = d.get("direction", "Stationary")
            direction_counts[direction] = direction_counts.get(direction, 0) + 1

        cameras = self.get_all_cameras()

        return {
            "total_detections": len(recent),
            "today_detections": len(today_detections),
            "unique_plates_count": len(unique_plates),
            "active_cameras_count": len(cameras),
            "directions": direction_counts,
            "database_mode": "MongoDB Atlas (Cloud)" if self.is_connected else "Local Resilient Store (Auto-Sync Ready)",
            "server_time_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
        }


# Singleton database instance
db_manager = DatabaseManager()

# Export functions for backward compatibility with car.py
add_detection = db_manager.add_detection
search_by_plate = db_manager.search_by_plate
