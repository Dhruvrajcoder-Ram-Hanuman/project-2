# 🛡️ SENTINEL ANPR - Vehicle Intelligence & Route Tracking Dashboard

> **Enterprise-Grade Automatic Number Plate Recognition (ANPR / ALPR) & Vehicle Route Reconstruction System**  
> Powered by **YOLOv8**, **EasyOCR**, **FastAPI**, **Leaflet.js Maps**, and **MongoDB Atlas** (with Resilient Local Failover).

---

## 🌟 Overview

SENTINEL ANPR is a complete, full-stack vehicle tracking surveillance command center. It analyzes traffic camera feeds in real time, tracks vehicle motion and direction, detects and extracts license plate text, records GPS-tagged sightings, and reconstructs chronological journeys across city camera networks on an interactive map.

---

## 🚀 Key Features

1. **Live Video Surveillance & ANPR Feed**
   - Real-time MJPEG browser streaming (`/api/video_feed`).
   - Dynamic bounding box tracking with YOLOv8.
   - Live travel direction calculation (Northbound, Southbound, Eastbound, Westbound, Stationary).
   - High-tech tactical HUD overlay showing camera node, street address, and live GPS coordinates.
   - Built-in traffic simulation mode when physical webcams are offline or in headless environments.

2. **Vehicle Route Reconstruction & Interactive Map**
   - Search any vehicle registration plate (e.g. `HR06A1245`).
   - Reconstructs chronological travel path across Panipat/city camera nodes.
   - Visualized on an interactive **Leaflet.js** map with custom glowing markers and directional polyline arrows.
   - Step-by-step chronological investigation timeline.

3. **Multi-Camera Node Network**
   - Multi-camera registry covering strategic corridors (Panipat Toll Plaza NH-44, GT Road Bus Stand, Sector 18 Main Gate).
   - Real-time telemetry, uptime status, and detection counters per node.

4. **Historical Database & Records**
   - Filterable, searchable datatable with confidence scores, timestamps, and camera tags.
   - One-click CSV export for official surveillance reports.

5. **Investigation Studio (Image ANPR Scanner)**
   - Upload any vehicle photo to extract plates on demand.
   - Highlights detected bounding box and confidence score with direct option to track the vehicle's history.

6. **Enterprise Dual-Layer Database Architecture**
   - Connects to **MongoDB Atlas** (`Car.vehicle` collection) using atomic `$push` operations for high concurrency.
   - Features **Resilient Local Failover**: If MongoDB Atlas has SSL issues or IP whitelisting restrictions, the system operates uninterrupted using a local mirror (`data/local_storage.json`).

---

## 📁 Project Architecture

```
project-2-main/
│
├── server.py              # FastAPI Web Server, REST APIs & MJPEG Streaming
├── database.py            # Unified Database Manager (MongoDB Atlas + Resilient Local Store)
├── camera_stream.py       # Computer Vision Pipeline (YOLOv8 + EasyOCR + Camera Streamer)
├── camara.py              # Standalone OpenCV / ANPR Script
├── camera.py              # English-spelling alias wrapper for camara.py
├── car.py                 # Core vehicle detection models & search logic
│
├── static/
│   └── index.html         # High-End Dark Tactical SOC Web Dashboard (Tailwind + Leaflet)
│
├── data/
│   └── local_storage.json # Resilient local database mirror
│
├── run_dashboard.bat      # One-click Windows batch launcher
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

---

## ⚡ Quick Start

### 1. Launch with One Click (Windows)
Double-click `run_dashboard.bat` in this folder.

### 2. Or Launch via Terminal
```bash
python server.py
```

Open your browser at:
👉 **`http://localhost:8000`**

---

## 📡 REST API Documentation

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the ANPR Command Center Web Dashboard |
| `GET` | `/api/status` | System health, camera status, and database connectivity |
| `GET` | `/api/stats` | Aggregated analytics (total sightings, unique plates, today's count) |
| `GET` | `/api/cameras` | List of registered camera nodes and coordinates |
| `GET` | `/api/recent` | Recent vehicle detection feed |
| `GET` | `/api/search?plate={plate}` | Vehicle route history & chronological checkpoints |
| `GET` | `/api/video_feed` | Live MJPEG video stream with ANPR HUD overlay |
| `POST` | `/api/scan_image` | Upload image for instant plate extraction |
| `POST` | `/api/camera/switch` | Switch active camera node |
| `POST` | `/api/test_detection` | Inject demo detection event |

Interactive Swagger API docs available at: **`http://localhost:8000/docs`**
