---
title: Sentinel ANPR
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# 🛡️ SENTINEL ANPR - City Surveillance & Vehicle Route Intelligence System

> **Enterprise-Grade Automatic Number Plate Recognition (ANPR / ALPR) & Vehicle Route Reconstruction Command Center**  
> Built with **FastAPI**, **YOLOv8**, **EasyOCR**, **OpenCV**, **Leaflet.js**, and **MongoDB Atlas** (with Resilient Local Failover).

---

## 🌟 Quick Overview / Project Highlights

SENTINEL ANPR is a complete, full-stack vehicle tracking surveillance command center designed for police, traffic authorities, and smart city infrastructure:
- 📹 **4-Camera Live City Surveillance Grid**: Multi-camera CCTV layout with independent road perspectives, vehicle animations, and ANPR telemetry.
- 🔄 **City Simulation vs Physical Webcam**: Toggle between simulated high-density traffic feeds or live physical webcams with real-time YOLOv8 vehicle tracking and EasyOCR plate extraction.
- 🗺️ **Vehicle Route Reconstruction**: Search any vehicle plate number (e.g. `HR06A1245`) to plot its chronological route across all city camera checkpoints on an interactive map.
- 🔍 **On-Demand Image ANPR Scanner**: Upload vehicle images to extract plates and bounding boxes instantly.
- 📋 **Full Surveillance Database & CSV Export**: Chronological logs with confidence scores, direction analysis, and exportable surveillance reports.
- 🛡️ **Zero-Downtime Dual Storage**: Connects to MongoDB Atlas when online, with instant automatic failover to local resilient JSON storage (`data/local_storage.json`).

---

## 🚀 Step-by-Step Run Guide (Zero Se Start Karein)

Neeche diye gaye steps follow karke project ko kisi bhi system par 0 se run kar sakte hain:

### 📋 Prerequisites (Zaroori Cheezein)
1. **Python 3.10 ya Python 3.11** (Recommended: Python 3.11)
   - [Python Download Link](https://www.python.org/downloads/)
   - *Note: Python install karte waqt **"Add Python to PATH"** checkbox zaroor check karein!*
2. **Git** (Code clone karne ke liye)
   - [Git Download Link](https://git-scm.com/downloads)

---

### Step 1: Clone the Repository
Apne terminal ya command prompt mein yeh command chalayein:

```bash
git clone https://github.com/Dhruvrajcoder-Ram-Hanuman/project-2.git
cd project-2
```

---

### Step 2: (Optional but Recommended) Virtual Environment Banayein

#### Windows (Command Prompt / PowerShell):
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3: Dependencies Install Karein
Saare required libraries (FastAPI, Ultralytics YOLOv8, EasyOCR, OpenCV, PyMongo, etc.) install karne ke liye run karein:

```bash
pip install -r requirements.txt
```

---

### Step 4: Run the Project (Server Start Karein)

#### Option A: One-Click Launcher (Windows ke liye Sabse Aasan)
Sirf **`run_dashboard.bat`** file par double-click karein!  
Yeh automatically Python detect karega, server start karega aur browser mein website open kar dega.

#### Option B: Terminal Command se Run Karein
```bash
python server.py
```

*Agar aapke system par multiple Python versions hain (jaise Python 3.11):*
```powershell
& "C:\Users\<YourUser>\AppData\Local\Programs\Python\Python311\python.exe" server.py
```

---

### Step 5: Dashboard Open Karein
Server start hone ke baad browser mein yeh URL kholein:

👉 **`http://localhost:8000`**

Interactive API Docs (Swagger UI) dekhne ke liye:  
👉 **`http://localhost:8000/docs`**

---

## ☁️ 100% Free Cloud Deployment Guide (Render.com)

Aap is project ko **Render.com** par 100% free live host kar sakte hain taaki koi bhi internet se direct access kar sake:

1. **Render par Account Banayein:**
   - [https://render.com](https://render.com) par jayein aur **"Sign In with GitHub"** karein.
2. **New Web Service Create Karein:**
   - Dashboard mein **`New +`** button dabakar **`Web Service`** chunein.
   - Apni repository **`Dhruvrajcoder-Ram-Hanuman/project-2`** ko connect karein.
3. **Settings (render.yaml se auto-detect ho jayengi):**
   - **Name:** `sentinel-anpr`
   - **Region:** Singapore ya Oregon
   - **Branch:** `main`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn server:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** `Free`
4. **Deploy Web Service:**
   - **`Deploy Web Service`** par click karein.
   - 2 se 3 minute ke andar aapka project live ho jayega aur aapko ek permanent public URL mil jayega:  
     👉 `https://sentinel-anpr.onrender.com`

---

## 🤗 Hugging Face Spaces Deployment Guide (AI ke liye 16GB RAM FREE)

Hugging Face Spaces AI & Computer Vision projects ke liye duniya ka sabse popular platform hai:

1. **Hugging Face Account Banayein:**
   - [https://huggingface.co](https://huggingface.co) par sign up / login karein.
2. **New Space Create Karein:**
   - Top-right profile icon par click karein aur **"New Space"** chunein.
   - **Space Name:** `sentinel-anpr`
   - **License:** `mit` ya `apache-2.0`
   - **Space SDK:** **`Docker`** *(Blank chunein)*
   - **Space Hardware:** **`CPU basic · 2 vCPU · 16 GB RAM · FREE`**
   - **Visibility:** `Public`
   - Click **"Create Space"**.
3. **Repository Connect / Push Karein:**
   - Space banne ke baad Hugging Face aapko Git clone link dega.
   - Apne computer terminal par run karein:
   ```bash
   git remote add hf https://huggingface.co/spaces/<AapkaUsername>/sentinel-anpr
   git push hf main
   ```
   *(Ya fir Hugging Face Space settings mein jakar "Connect GitHub repository" se direct connect kar sakte hain).*
4. **Live URL:**
   - 2 minute mein build complete ho jayega aur aapka permanent free AI dashboard live ho jayega:  
     👉 `https://<AapkaUsername>-sentinel-anpr.hf.space`

---

## 🎛️ Dashboard Features & How to Use

### 1. 📹 Live Surveillance Grid
- **4 City Cameras**:
  - `CAM_001`: Panipat Toll Plaza (NH-44 Highway)
  - `CAM_004`: GT Road, Bus Stand Chowk (High-Density City Traffic)
  - `CAM_005`: Sector 18 Main Gate (City Suburb Entrance)
  - `CAM_002`: Ibrahim Lodi Road, Model Town (Commercial Center)
- **Layout Switcher**: Switch between `2x2 Grid (⊞)`, `Focus View (⬛◻)`, `3-Wide (≣)`, and `Single View (□)`.
- **Camera Filters**: One-click pills to isolate any camera.
- **⛶ Expand Modal**: Click Expand on any feed to view it cinema-style.
- **🏙️ City Sim / 📹 Webcam Toggle**: Switch between realistic city simulation and physical webcam detection.

### 2. 🗺️ Vehicle Route Tracker
- Top search bar ya quick chips se koi bhi number plate search karein (e.g. `HR06A1245`, `DL10CB4921`, `UP16BE7789`, `CH01AB9988`).
- Panipat city map par car ka poora chronological safar timeline aur route line ke saath display ho jayega.

### 3. 🔍 Image Scanner
- Kisi bhi vehicle ki photo drag & drop karein ya upload karein.
- AI engine (YOLOv8 + EasyOCR) plate detect karke license plate number aur confidence score nikalega aur directly database mein log karega.

### 4. 📋 Records & CSV Export
- Sabhi detections table mein timestamp, camera, location aur confidence score ke saath available hain.
- **`⬇ CSV`** button dabakar poori report Excel / CSV format mein download karein.

---

## 📁 Project Directory Structure

```
project-2/
│
├── server.py              # FastAPI server, REST API endpoints & MJPEG multi-feed streamer
├── camera_stream.py       # YOLOv8 + EasyOCR vision pipeline & city surveillance rendering engine
├── database.py            # Unified Database Manager (MongoDB Atlas + resilient local JSON store)
├── car.py                 # Core vehicle tracking & detection service
├── camera.py              # Camera wrapper module
├── camara.py              # Legacy standalone OpenCV vision script
│
├── static/
│   └── index.html         # Pure CSS/JS dark tactical CCTV Command Center dashboard (Zero-CDN)
│
├── data/
│   └── local_storage.json # Offline database mirror (pre-seeded with Panipat camera checkpoints)
│
├── yolov8n.pt             # Pre-trained YOLOv8 Nano weights for vehicle detection
├── run_dashboard.bat      # One-click Windows dashboard runner & browser launcher
├── requirements.txt       # Python dependencies list
├── .gitignore             # Git ignore patterns
└── README.md              # Complete documentation and setup guide
```

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Command Center Web Dashboard HTML |
| `GET` | `/api/status` | System health, database connectivity, active mode |
| `GET` | `/api/stats` | Analytics summary (total detections, unique plates, active cameras) |
| `GET` | `/api/cameras` | List of all registered city camera nodes |
| `GET` | `/api/recent` | Recent vehicle sightings feed |
| `GET` | `/api/search?plate={plate}` | Vehicle journey & chronological route history |
| `GET` | `/api/feed/{cam_id}` | Real-time MJPEG live stream for a specific camera |
| `GET` | `/api/video_feed` | Primary camera MJPEG stream |
| `POST` | `/api/scan_image` | Upload image for instant plate recognition |
| `POST` | `/api/camera/mode` | Switch feed mode (`simulation` or `webcam`) |
| `POST` | `/api/camera/switch` | Switch active camera node |
| `POST` | `/api/test_detection` | Inject a mock detection event |

---

## ❓ Troubleshooting & FAQs

### Q1: Camera screen black kyu dikh rahi thi pehle?
**Answer:** Windows Hello IR camera (Index 0) black frames return karta hai agar laptop cover ho ya IR mode mein ho. Ab system mein **Smart Auto-Detect** laga hai jo automatically non-zero working camera pick karta hai, aur dashboard default mein **City Simulation Mode** par chalta hai taaki bina camera ke bhi realistic traffic surveillance chalta rahe.

### Q2: MongoDB Atlas connect nahi ho raha to kya project chalega?
**Answer:** Haan! System mein **Resilient Local Storage Fallback** hai. Agar MongoDB Atlas SSL handshake ya internet issue ki wajah se unavailable ho, to saara data bina rukavat ke `data/local_storage.json` mein save aur search hota hai.

### Q3: Python version error aaye to kya karein?
**Answer:** EasyOCR aur PyTorch **Python 3.10 ya 3.11** par best chalte hain. Agar system default Python 3.14 ya koi naya version hai, to Python 3.11 install karke us path se `server.py` run karein.

---

## 👥 Contributors & License
Developed for Automatic Number Plate Recognition and Smart City Surveillance.  
Open for academic and security surveillance development.
