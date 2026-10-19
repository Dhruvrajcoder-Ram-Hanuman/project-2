from datetime import datetime, timedelta, timezone
import json
import re
from pydantic import BaseModel
from pymongo import MongoClient

# MongoDB Atlas Cluster URI
MONGO_URI = (
    "mongodb+srv://vercelweb1_db_user:DhruvrajGoyal@cluster0.mi37f6m.mongodb.net/"
)
client = MongoClient(MONGO_URI)

# Database: "Car" | Collection: "vehicle"
db = client["Car"]
collection = db["vehicle"]


class VehicleDetection(BaseModel):
    location: str
    latitude: float
    longitude: float
    direction: str
    timestamp: str
    confidence: float
    status: str


IST = timezone(timedelta(hours=5, minutes=30))


def add_detection(
    cam,
    plate,
    *,
    location,
    latitude,
    longitude,
    direction,
    timestamp=None,
    confidence,
    status,
):
    """Takes existing camera data from MongoDB, appends the new detection, and updates MongoDB."""
    if timestamp is None:
        timestamp = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

    clean_plate = re.sub(r"[^A-Z0-9]", "", plate.upper())

    det = VehicleDetection(
        location=location,
        latitude=latitude,
        longitude=longitude,
        direction=direction,
        timestamp=timestamp,
        confidence=confidence,
        status=status,
    )
    record = det.model_dump(mode="json")

    # 1. TAKE existing camera document from MongoDB (or create a new structure if not found)
    cam_doc = collection.find_one({"_id": cam})
    if cam_doc is None:
        cam_doc = {"_id": cam, "plates": {}}

    # 2. UPDATE the dictionary in Python (same logic as setdefault on df.json)
    cam_doc.setdefault("plates", {}).setdefault(clean_plate, []).append(record)

    # 3. SAVE updated document back into MongoDB
    collection.replace_one({"_id": cam}, cam_doc, upsert=True)
    return cam_doc


def search_by_plate(plate: str):
    """Takes a car number plate and searches its detection details across every camera in MongoDB."""
    clean_plate = re.sub(r"[^A-Z0-9]", "", plate.upper())
    if not clean_plate:
        return {
            "plate_number": "",
            "total_sightings": 0,
            "cameras_spotted": [],
            "history": [],
        }

    # Find every camera document in Car.vehicle where this plate number exists
    cursor = collection.find({f"plates.{clean_plate}": {"$exists": True}})

    sightings = []
    cameras_spotted = []

    for doc in cursor:
        cam_id = doc["_id"]
        cameras_spotted.append(cam_id)
        plate_records = doc.get("plates", {}).get(clean_plate, [])

        for rec in plate_records:
            sightings.append({
                "camera": cam_id,
                "location": rec.get("location"),
                "latitude": rec.get("latitude"),
                "longitude": rec.get("longitude"),
                "direction": rec.get("direction"),
                "timestamp": rec.get("timestamp"),
                "confidence": rec.get("confidence"),
                "status": rec.get("status"),
            })

    # Sort all sightings chronologically by timestamp
    sightings.sort(key=lambda x: x.get("timestamp", ""))

    return {
        "plate_number": clean_plate,
        "total_sightings": len(sightings),
        "cameras_spotted": cameras_spotted,
        "history": sightings,
    }


if __name__ == "__main__":
    # 1. Test taking & updating data in MongoDB
    add_detection(
        "CAM_005",
        "HR06A1245",
        location="GT Road, Panipat",
        latitude=29.3909,
        longitude=76.9635,
        direction="Northbound",
        timestamp=datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        confidence=0.968,
        status="high_confidence",
    )
    print("Detection saved to MongoDB.\n")

    # 2. Test searching a vehicle plate across all cameras
    car_number = input(
        "Enter car number plate to search (or press Enter for HR06A1245): "
    ).strip()
    if not car_number:
        car_number = "HR06A1245"

    result = search_by_plate(car_number)
    print(json.dumps(result, indent=2))
