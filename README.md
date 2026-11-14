# 🛡️ CrowdShield Pro

**AI-powered crowd management and stampede prevention system** built with YOLOv8 + SORT tracking + Streamlit.

---

## Features

- **Real-time crowd detection** using YOLOv8s object detection
- **Multi-object tracking** with SORT (Kalman filter + Hungarian algorithm)
- **Thermal density heatmap** overlaid on live feed
- **Auto night-vision** via Gamma Correction + CLAHE for low-light scenes
- **GDPR-compliant face blurring** (top 20% of each detected bounding box)
- **Dynamic alert levels** — SAFE → LOW/MODERATE/HIGH EMERGENCY
- **Emergency SMS alerts** via Twilio (async, throttled)
- **CAP v1.2 protocol** XML dispatch for government/enterprise integration
- **Incident log** with auto-snapshot capture stored in SQLite
- **Multiple input sources**: Webcam, CCTV/RTSP, Video upload, Static image
- **Streamlit dashboard** with dark-mode professional UI

---

## Project Structure

```
Crowd_Shield/
├── crowd-dashboard/
│   ├── app.py              # Main Streamlit application
│   ├── tracker.py          # SORT multi-object tracker
│   ├── heatmap.py          # Adaptive thermal heatmap engine
│   ├── database.py         # SQLite CRUD helpers
│   ├── twilio_alert.py     # Async SMS alert dispatcher
│   ├── cap_alert.py        # CAP v1.2 XML alert generator
│   ├── requirements.txt    # Python dependencies
│   ├── .env.example        # Environment variable template
│   └── CS LOGO.png         # App logo
└── README.md
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Crowd_Shield.git
cd Crowd_Shield/crowd-dashboard
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Edit .env and fill in your Twilio credentials
```

### 5. Download YOLO model weights
The app uses `yolov8s.pt` by default (placed in the `crowd-dashboard/` folder).
```bash
# Automatically downloaded by ultralytics on first run, or manually:
# https://github.com/ultralytics/assets/releases
```

### 6. Run the app
```bash
streamlit run app.py
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `TWILIO_ACCOUNT_SID` | Twilio Account SID (from Twilio Console) |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token |
| `TWILIO_PHONE_NUMBER` | Your Twilio phone number (E.164 format) |
| `TWILIO_DEST_PHONE` | Destination phone for emergency alerts |
| `CAP_DISPATCH_ENDPOINT` | *(Optional)* Webhook URL for CAP XML alerts |

> **Never commit your `.env` file.** Use `.env.example` as a template.

---

## Alert Levels

| Ratio (Count / Capacity) | Status | Action |
|---|---|---|
| ≤ 1.1× | 🟢 SAFE | Monitor normally |
| 1.1× – 1.5× | 🟡 LOW EMERGENCY | Restrict entry |
| 1.5× – 2.0× | 🟠 MODERATE EMERGENCY | Begin dispersal |
| > 2.0× | 🔴 HIGH EMERGENCY | Immediate evacuation + CAP alert |

---

## Tech Stack

- **Detection**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Tracking**: SORT (Simple Online Realtime Tracking)
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Database**: SQLite (via Python `sqlite3`)
- **Alerts**: [Twilio SMS API](https://www.twilio.com/)
- **Protocol**: OASIS CAP v1.2 XML

---

## License

MIT License — see [LICENSE](LICENSE) for details.
