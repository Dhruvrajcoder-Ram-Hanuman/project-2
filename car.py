from datetime import datetime, timedelta, timezone
import json
import re
from pydantic import BaseModel

# Import resilient database manager
from database import IST, VehicleDetection, add_detection, db_manager, search_by_plate


if __name__ == "__main__":
    print("=" * 60)
    print("  ANPR Surveillance Database Service (car.py)")
    print("=" * 60)

    # 1. Test adding detection
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
    print("\n[+] Detection record processed & saved.")

    # 2. Test searching vehicle plate across all cameras
    car_number = input(
        "\nEnter car number plate to search (or press Enter for HR06A1245): "
    ).strip()
    if not car_number:
        car_number = "HR06A1245"

    result = search_by_plate(car_number)
    print("\n--- Search Results ---")
    print(json.dumps(result, indent=2))
