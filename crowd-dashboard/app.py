import os
import cv2
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import pandas as pd
import numpy as np
import time
import tempfile
import re
import uuid
from ultralytics import YOLO
from heatmap import Heatmap
from tracker import SORT
from database import create_tables, add_user, check_user, insert_crowd_log, insert_alert, insert_emergency_snapshot, get_emergency_snapshots, delete_all_snapshots
from datetime import datetime
import random
from streamlit_js_eval import streamlit_js_eval
from twilio_alert import send_emergency_sms_async
from cap_alert import dispatch_cap_alert
# ---------------- DATABASE INITIALIZATION ----------------
create_tables()

# ---------------- SNAPSHOT DIRECTORY ----------------
SNAPSHOT_DIR = "snapshots"
if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)

# ---------------- PASSWORD VALIDATION ----------------
def is_strong_password(password):
    if len(password) < 8: return False
    if not re.search(r"[A-Z]", password): return False
    if not re.search(r"[a-z]", password): return False
    if not re.search(r"[0-9]", password): return False
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password): return False
    return True

# ---------------- SESSION & AUTH PERSISTENCE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Session always starts unauthenticated — login required every time

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"

# Camera session state
if "cam_tracker" not in st.session_state:
    st.session_state.cam_tracker  = None
if "cam_heatmap" not in st.session_state:
    st.session_state.cam_heatmap  = None
if "cam_graph" not in st.session_state:
    st.session_state.cam_graph    = []
if "cam_peak" not in st.session_state:
    st.session_state.cam_peak     = 0
if "cam_running" not in st.session_state:
    st.session_state.cam_running  = False
if "last_sms_time" not in st.session_state:
    st.session_state.last_sms_time = 0

# --- 1. CORE SYSTEM CONFIGURATION ---
st.set_page_config(
    layout="wide", 
    page_title="CrowdShield Pro | Command Center",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# --- 2. PRO UI/UX STYLING (HACKATHON ELITE) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Overrides */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    /* Professional Headings */
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        font-weight: 600 !important;
        letter-spacing: -0.025em !important;
        color: #f8fafc !important;
        text-transform: none;
        text-shadow: none;
    }

    /* Generic Card / Block */
    [data-testid="stVerticalBlock"] > div > div {
        border-radius: 8px;
    }

    /* Metric Refinement */
    [data-testid="stMetric"] {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        transform: none;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
        border-color: #3b82f6 !important;
    }
    [data-testid="stMetricLabel"] { 
        color: #94a3b8 !important; 
        font-family: 'Inter', sans-serif !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        text-shadow: none;
    }
    [data-testid="stMetricValue"] { 
        color: #f8fafc !important; 
        font-family: 'Inter', sans-serif !important;
        font-size: 2.25rem !important; 
        font-weight: 700 !important;
        text-shadow: none;
    }

    /* Status Banner */
    @keyframes subtlePulse {
        0% { box-shadow: 0 0 0 0 rgba(opacity, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(opacity, 0); }
        100% { box-shadow: 0 0 0 0 rgba(opacity, 0); }
    }
    .status-banner {
        padding: 16px 24px;
        background: #1e293b;
        border: 1px solid #334155;
        border-left: 4px solid;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 1.125rem;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: #f8fafc;
    }

    /* Buttons */
    .stButton>button {
        background: #3b82f6;
        color: #ffffff !important;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        letter-spacing: normal;
        transition: all 0.2s ease;
        text-transform: none;
        width: 100%;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        text-shadow: none;
    }
    .stButton>button:hover {
        background: #2563eb;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        color: #ffffff !important;
    }
    .stButton>button:active {
        background: #1d4ed8;
        transform: translateY(0);
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    .stButton>button[disabled] {
        background: #475569 !important;
        color: #94a3b8 !important;
        cursor: not-allowed;
        transform: none;
        box-shadow: none;
    }

    /* Custom CSS for Login Card Box */
    [data-testid="stForm"] {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
        padding: 32px !important;
        margin-top: 10px;
        border-radius: 12px !important;
    }

    /* Inputs */
    .stTextInput>div>div>input {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        font-weight: 400;
        box-shadow: none;
        padding: 0.5rem 0.75rem !important;
        transition: border-color 0.2s ease;
    }
    .stTextInput>div>div>input:focus {
        border-color: #3b82f6 !important;
        outline: none;
    }
    /* Streamlit labels inside text inputs */
    .stTextInput label p {
        color: #cbd5e1 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 4px;
        text-transform: none;
    }
    
    #MainMenu, footer { visibility: hidden; }
    
    /* Chart Styling */
    .stAreaChart {
        background-color: #1e293b;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        padding: 16px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
        border-bottom: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab"] {
        height: auto;
        white-space: pre-wrap;
        background-color: transparent;
        border: none;
        border-radius: 0;
        padding: 12px 16px;
        margin-top: 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        font-weight: 500;
        color: #94a3b8;
        transition: all 0.2s;
        text-transform: none;
        border-bottom: 2px solid transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: transparent;
        color: #f8fafc;
        margin-top: 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: #3b82f6 !important;
        border-bottom: 2px solid #3b82f6 !important;
        margin-top: 0;
        padding-bottom: 12px;
        text-shadow: none;
    }
    
    /* Incident Logs */
    div[data-testid="stVerticalBlock"] > div > div > div > div > div[style*="background: #ffffff"] {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Selectboxes and Sliders overrides */
    div[data-baseweb="select"] > div {
        background-color: #0f172a !important;
        border-color: #334155 !important;
    }
    div[data-baseweb="select"] > div * {
        color: #f8fafc !important;
    }

    </style>
    """, unsafe_allow_html=True)

# ---------------- AUTH PAGE ----------------
def auth_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col2, _ = st.columns([1, 2, 1])
    
    with col2:
        
        # Using columns to center the image perfectly above the text
        img_col1, img_col2, img_col3 = st.columns([1, 1, 1])
        with img_col2:
            try:
                logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CS LOGO.png")
                st.image(logo_path)
            except Exception:
                st.markdown("<h1 style='text-align: center; color: #f8fafc; margin-bottom: 5px; font-weight: 700;'>🛡️ CrowdShield Pro</h1>", unsafe_allow_html=True)
                
        st.markdown("<p style='text-align: center; color: #94a3b8; margin-bottom: 30px; font-size: 0.875rem;'>Advanced Live Analytics Command Center</p>", unsafe_allow_html=True)
        
        if st.session_state.auth_mode == "login":
            st.markdown("<h3 style='margin-bottom: 20px; text-align: center; color: #f8fafc;'>Command Access</h3>", unsafe_allow_html=True)
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Identifier")
                password = st.text_input("Access Key", type="password")
                if st.form_submit_button("Authorize Session"):
                    if check_user(username, password):
                        st.session_state.logged_in = True
                        # Persist to localStorage via JS
                        streamlit_js_eval(js_expressions="localStorage.setItem('crowdshield_auth', 'true')", key="set_auth")
                        st.rerun()
                    else:
                        st.error("Access Denied: Invalid Credentials")
            
            if st.button("New Agent Registration"):
                st.session_state.auth_mode = "signup"
                st.rerun()
        else:
            st.markdown("<h3 style='margin-bottom: 20px; text-align: center; color: #f8fafc;'>Provision Agent</h3>", unsafe_allow_html=True)
            with st.form("signup_form"):
                new_user = st.text_input("Assign ID")
                new_pass = st.text_input("Create Passkey", type="password")
                new_email = st.text_input("Contact Email")
                if st.form_submit_button("Enroll System"):
                    if is_strong_password(new_pass):
                        if add_user(new_user, new_pass, new_email):
                            st.success("Account Provisioned. Proceed to Security Check.")
                            st.session_state.auth_mode = "login"
                            st.rerun()
                        else:
                            st.error("ID Already Exists in Database")
                    else: 
                        st.error("Weak Passkey. Use mix of A-Z, 0-9, and symbols.")
            if st.button("Back to Terminal"):
                st.session_state.auth_mode = "login"
                st.rerun()

if not st.session_state.logged_in:
    auth_page()
    st.stop()

# --- 4. RESOURCE LOADING ---
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")

model = load_model()

# --- 5. DYNAMIC STATUS LOGIC ---
def get_dynamic_status(count, capacity):
    if capacity <= 0:
        return "#808080", "CALIBRATING", "rgba(128, 128, 128, 0.1)"
    
    ratio = count / capacity

    if ratio > 2.0:
        return "#7d0000", "HIGH EMERGENCY", "rgba(125, 0, 0, 0.1)"
    elif ratio > 1.5:
        return "#ff4b4b", "MODERATE EMERGENCY", "rgba(255, 75, 75, 0.05)"
    elif ratio > 1.1:
        return "#ffbf00", "LOW EMERGENCY", "rgba(255, 191, 0, 0.05)"
    else:
        return "#00ff7f", "SAFE", "rgba(0, 255, 127, 0.05)"

def calculate_emergency_level(capacity_limit, current_occupancy):
    """
    Calculates the emergency level based on current occupancy compared to capacity limit.
    Returns a dictionary (JSON object) containing the alert level and a custom warning message.
    """
    if capacity_limit <= 0:
        return {
            "alert_level": "Error",
            "message": "Capacity limit must be greater than zero."
        }

    ratio = current_occupancy / capacity_limit

    if ratio > 2.0:
        return {
            "alert_level": "High Emergency",
            "message": "CRITICAL DANGER: Occupancy is more than double the allowed limit! Immediate evacuation and crowd control required."
        }
    elif ratio > 1.5:
        return {
            "alert_level": "Moderate Emergency",
            "message": "WARNING: Occupancy is over 1.5x the limit. Please restrict entry and begin dispersing the crowd."
        }
    elif ratio > 1.1:
        return {
            "alert_level": "Low Emergency",
            "message": "ALERT: Occupancy is slightly above the limit. Monitor the situation closely."
        }
    else:
        return {
            "alert_level": "Normal",
            "message": "Occupancy is within safe limits."
        }

# --- 6. CORE IMAGE PROCESSING FUNCTION ---
def enhance_image(frame):
    """
    Auto Night Vision: Measures the mean luminosity of the scene.
    If the frame is determined to be genuinely dark (mean L < 80),
    the system self-activates Gamma Correction + CLAHE to extract
    human silhouettes from shadows before passing to YOLO.
    """
    # Measure mean brightness using LAB lightness channel (perceptually accurate)
    lab_check = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    mean_brightness = cv2.mean(lab_check)[0]  # L channel mean (0-255)
    
    LOW_LIGHT_THRESHOLD = 80  # Below this = night-time / dark environment
    
    if mean_brightness >= LOW_LIGHT_THRESHOLD:
        return frame  # Well-lit scene: fast path, no processing overhead
    
    # --- AUTO NIGHT VISION ACTIVE ---
    # 1. Gamma Correction to non-linearly brighten pitch-black shadows
    gamma = 2.0
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    brightened = cv2.LUT(frame, table)

    # 2. Aggressive CLAHE on L-channel to separate silhouettes from background
    lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced

def process_frame(frame, tracker, heatmap, conf, nms, run_ai=True):
    # Auto-enhanced preprocessing: night vision fires only when the scene is dark
    enhanced_frame = enhance_image(frame)
    
    if run_ai:
        # Run prediction for real-time speed while maximizing accuracy for distant crowds.
        # We process the frame at 1280px and increase max_det to detect distant people and massive crowds.
        results = model.predict(enhanced_frame, conf=conf, iou=nms, imgsz=1280, max_det=2000, classes=[0], augment=True, agnostic_nms=True, verbose=False)[0]
        
        raw_dets = []
        for box in results.boxes:
            if int(box.cls) == 0:
                conf_score = float(box.conf)
                # Use dynamic confidence threshold strictly avoiding hardcoded minimums
                if conf_score < conf: continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                raw_dets.append([x1, y1, x2, y2])

        active_tracks = tracker.update(raw_dets)
        
        # Cache tracks globally so intermediate smoothly-playing frames can render them without computing AI
        if 'last_active_tracks' not in st.session_state:
            st.session_state.last_active_tracks = active_tracks
        else:
            st.session_state.last_active_tracks = active_tracks
            
        live_count    = len(active_tracks)
        total_unique  = tracker.total_unique
        st.session_state.last_live_count = live_count
        st.session_state.last_total_unique = total_unique
    else:
        # For intermediate frames (to maintain smooth 30 FPS playback), we simply reuse the last known detections
        # without invoking the heavy YOLO neural network or forcing the tracker to step its Kalman filter.
        active_tracks = st.session_state.get('last_active_tracks', [])
        live_count = st.session_state.get('last_live_count', 0)
        total_unique = st.session_state.get('last_total_unique', 0)

    # Collect foot-point + person height for the adaptive heatmap
    heat_points = []  # (foot_x, foot_y, person_height_px)

    # Fixed visual styling for all IDs (Spring Green BGR: 127, 255, 0)
    box_color  = (255, 144, 30)  # Modern Blue/Cyan BGR
    text_color = (255, 255, 255)
    bg_color   = (255, 144, 30)
    thickness  = 2
    font_scale = 0.5
    font       = cv2.FONT_HERSHEY_SIMPLEX

    # Pass 1: GDPR Face Blur for all active tracks first
    # This ensures that overlapping faces do not blur the bounding boxes and text of people behind them
    for tr in active_tracks:
        x1, y1, x2, y2 = tr['bbox']
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Apply strict GDPR Face Blur directly to the top 20% of the detection box
        head_h = int((y2 - y1) * 0.20)
        if head_h > 0 and (x2 - x1) > 0:
            roi = frame[y1:y1+head_h, x1:x2]
            frame[y1:y1+head_h, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 30)

    # Pass 2: Draw the UI elements and collect foot points
    for tr in active_tracks:
        tid  = tr['id']
        x1, y1, x2, y2 = tr['bbox']
        
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Draw uniform bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
        
        label  = f"ID #{tid}"
        (lw, lh), _ = cv2.getTextSize(label, font, font_scale, thickness)
        lx, ly = x1, max(y1 - 4, lh + 4)
        
        # Draw uniform text background
        cv2.rectangle(frame, (lx, ly - lh - 4), (lx + lw + 6, ly + 2), bg_color, -1)
        
        # Draw uniform text
        cv2.putText(frame, label, (lx + 3, ly - 2), font, font_scale, text_color, thickness)
        
        # Heatmap anchor (centroid) instead of foot point prevents bottom-edge clipping (sunrise effect)
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        person_width_px = max(1, x2 - x1)
        person_height_px = max(1, y2 - y1)
        heat_points.append((center_x, center_y, person_width_px, person_height_px))

    heatmap.add_points(heat_points)
    heat_img = heatmap.get_heatmap(frame)

    overlay_text  = f"IN FRAME: {live_count}   |   TOTAL SEEN: {total_unique}"
    (ow, oh), _   = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(frame, overlay_text, (10, oh + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 150), 2)

    return frame, heat_img, live_count, total_unique

# --- 6. SIDEBAR COMPONENT ---
with st.sidebar:
    # Load Image Logo in Sidebar
    try:
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CS LOGO.png")
        st.image(logo_path)
    except Exception:
        # Fallback if image fails
        st.markdown("""
            <div style='text-align: center; margin-bottom: 30px;'>
                <h1 style='color: #3b82f6; font-size: 2.5rem; margin-bottom: 0;'>🛡️</h1>
                <h2 style='color: #f8fafc; font-size: 1.2rem; font-weight: 600; margin-top: 5px;'>CrowdShield Pro</h2>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
        <div style='text-align: center; margin-bottom: 30px;'>
            <p style='color: #64748b; font-size: 0.75rem;'>Secure Node v22.02</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='border: 0; border-top: 1px solid #334155; margin: 20px 0;'>", unsafe_allow_html=True)
    
    st.markdown("### Signal Control")
    
    if "last_source" not in st.session_state:
        st.session_state.last_source = "Camera"
        
    source = st.selectbox(
        "Signal Source",
        ["Camera", "CCTV", "Video", "Image"],
        key="source_select"
    )
    
    if source != st.session_state.last_source:
        st.session_state.last_source = source
        
        # Clear persistent UI states to wipe the dashboard screen
        keys_to_clear = [
            "last_status_html", "last_count", "last_label", "last_occ_pct", 
            "last_peak", "last_total", "last_frame", "last_heat", "last_graph"
        ]
        for k in keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]
                
        st.session_state.cam_running = False
        st.session_state.vid_playing = False
            
        st.rerun()

    img_file   = None
    video_file = None
    cctv_ip    = None

    if source == "Image":
        img_file = st.file_uploader("Upload Static Capture", type=["jpg", "jpeg", "png", "bmp"], key="image_upload")
    elif source == "Video":
        video_file = st.file_uploader("Upload Forensic Footage", type=["mp4", "avi", "mov"], key="video_upload")
    elif source == "CCTV":
        cctv_ip = st.text_input("CCTV IP / RTSP URL", key="cctv_ip_input")
    
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("⚙️  Engine Config"):
        area_capacity = st.number_input("Zone Capacity", min_value=1, value=20, key="capacity_input")
        st.markdown("<hr style='margin: 10px 0; border: 0; border-top: 1px solid #334155;'>", unsafe_allow_html=True)
        conf_thresh   = st.slider("Detection Confidence", min_value=0.01, max_value=1.0, value=0.05, step=0.01, key="conf_slider")
        nms_thresh    = st.slider("Overlap Sensitivity (NMS)", min_value=0.01, max_value=1.0, value=0.40, step=0.01, key="nms_slider")
    st.markdown("<div style='font-size: 0.75rem; color: #64748b; margin-top: 20px;'>System Engine: YOLOv8s + SORT<br>Status: Operational | Core: Local</div>", unsafe_allow_html=True)
    
    if st.button("Logout"):
        st.session_state.logged_in = False
        streamlit_js_eval(js_expressions="localStorage.removeItem('crowdshield_auth')", key="logout")
        st.rerun()

# --- 8. DASHBOARD LAYOUT ---
st.markdown("""
<div style='display: flex; align-items: center; gap: 12px; margin-bottom: 24px;'>
    <h1 style='margin: 0; font-size: 1.75rem;'>Dashboard Intelligence</h1>
    <span style='background: rgba(16, 185, 129, 0.1); color: #10b981; padding: 4px 12px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; border: 1px solid rgba(16, 185, 129, 0.2);'>
        ● LIVE
    </span>
</div>
""", unsafe_allow_html=True)

status_placeholder = st.empty()
if "last_status_html" in st.session_state:
    status_placeholder.markdown(st.session_state.last_status_html, unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
with m1: 
    count_metric = st.empty()
    if "last_count" in st.session_state: count_metric.metric("Live Count", st.session_state.last_count)
with m2: 
    status_metric = st.empty()
    if "last_label" in st.session_state: status_metric.metric("Safety", st.session_state.last_label)
with m3: 
    occupancy_metric = st.empty()
    if "last_occ_pct" in st.session_state: occupancy_metric.metric("Occupancy", st.session_state.last_occ_pct)
with m4: 
    peak_metric = st.empty()
    if "last_peak" in st.session_state: peak_metric.metric("Session Peak", st.session_state.last_peak)
with m5: 
    total_metric = st.empty()
    if "last_total" in st.session_state: total_metric.metric("Total Seen", st.session_state.last_total)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📡 Surveillance Feed", "📊 Data Intelligence", "📸 Incident Logs"])

with tab1:
    col_feed, col_heat = st.columns(2)
    with col_feed:
        st.markdown("<h4 style='color: #94a3b8 !important; font-size: 0.875rem; margin-bottom: 12px;'>Optical Detection</h4>", unsafe_allow_html=True)
        FRAME_WINDOW = st.empty()
        if "last_frame" in st.session_state: FRAME_WINDOW.image(st.session_state.last_frame)
    with col_heat:
        st.markdown("<h4 style='color: #94a3b8 !important; font-size: 0.875rem; margin-bottom: 12px;'>Thermal Density</h4>", unsafe_allow_html=True)
        HEAT_WINDOW = st.empty()
        if "last_heat" in st.session_state: HEAT_WINDOW.image(st.session_state.last_heat)

with tab2:
    st.markdown("<h4 style='color: #94a3b8 !important; font-size: 0.875rem; margin-bottom: 12px;'>Historical Signal Pattern</h4>", unsafe_allow_html=True)
    graph_placeholder = st.empty()
    if "last_graph" in st.session_state and len(st.session_state.last_graph) > 0:
        graph_placeholder.area_chart(pd.DataFrame(st.session_state.last_graph[-100:], columns=["People"]), height=150)

@st.fragment
def render_incident_logs():
    col_refresh, col_delete, _ = st.columns([1, 1.2, 4])
    with col_refresh:
        # Formally just a button: clicking it naturally reruns only this fragment!
        st.button("🔄 Refresh Logs", help="Pull latest incident snapshots autonomously")
    with col_delete:
        if st.button("🗑️ Wipe All Logs", type="secondary", help="Irreversibly cleans database metrics and local images"):
            paths_to_delete = delete_all_snapshots()
            for path in paths_to_delete:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        pass
            st.success("All analytical logs expunged.")
            time.sleep(1)
            st.rerun()
            
    snapshots = get_emergency_snapshots(limit=30)
    
    if not snapshots:
        st.markdown("<div style='text-align:center; padding: 40px; color: #94a3b8; border: 1px dashed #334155; border-radius: 8px;'>No incident snapshots recorded yet. Safe operations.</div>", unsafe_allow_html=True)
    else:
        cols = st.columns(3)
        for idx, snap in enumerate(snapshots):
            with cols[idx % 3]:
                st.markdown(f"""
                <div style='background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);'>
                    <div style='font-size: 0.75rem; color: #64748b; margin-bottom: 8px; font-family: monospace;'>ID: {snap['alert_id'][:8]}... | {snap['timestamp']}</div>
                    <div style='font-weight: 600; color: #ef4444; margin-bottom: 12px; font-size: 0.875rem;'>{snap['severity']} at {snap['location_id']}</div>
                """, unsafe_allow_html=True)
                
                if os.path.exists(snap['snapshot_path']):
                    st.image(snap['snapshot_path'], use_column_width="auto")
                else:
                    st.markdown("<div style='padding: 30px; text-align: center; border: 1px dashed #334155; color: #64748b; border-radius: 6px;'>IMAGE UNAVAILABLE</div>", unsafe_allow_html=True)
                    
                st.markdown(f"""
                    <div style='display: flex; justify-content: space-between; margin-top: 12px; font-size: 0.875rem; color: #cbd5e1;'>
                        <span>Threshold: <b style='color: #ef4444;'>{snap['capacity_limit']}</b></span>
                        <span>Detected: <b style='color: #f8fafc;'>{snap['current_count']}</b></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

with tab3:
    st.markdown("<h4 style='color: #94a3b8 !important; font-size: 0.875rem; margin-bottom: 12px;'>Recent Emergency Snapshots</h4>", unsafe_allow_html=True)
    render_incident_logs()

# --- Anti-Flicker Global Update Router ---
def cache_ui_state(status_html, count, label, occ_pct, peak, total, proc_f, proc_h, graph_arr):
    status_placeholder.markdown(status_html, unsafe_allow_html=True)
    count_metric.metric("Live Count", count)
    status_metric.metric("Safety", label)
    occupancy_metric.metric("Occupancy", f"{int(occ_pct)}%")
    peak_metric.metric("Session Peak", peak)
    total_metric.metric("Total Seen", total)
    
    img_f = cv2.cvtColor(proc_f, cv2.COLOR_BGR2RGB)
    img_h = cv2.cvtColor(proc_h, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(img_f)
    HEAT_WINDOW.image(img_h)
    
    if hasattr(graph_arr, 'append'):
        graph_placeholder.area_chart(pd.DataFrame(graph_arr[-100:], columns=["People"]), height=150)
        st.session_state.last_graph = list(graph_arr)
        
    st.session_state.last_status_html = status_html
    st.session_state.last_count = count
    st.session_state.last_label = label
    st.session_state.last_occ_pct = f"{int(occ_pct)}%"
    st.session_state.last_peak = peak
    st.session_state.last_total = total
    st.session_state.last_frame = img_f
    st.session_state.last_heat = img_h

# --- 9. EXECUTION ---
graph_data = []
max_count = 0

if source in ["Camera", "CCTV"]:
    ctrl_col1, ctrl_col2 = st.columns([1, 1])
    
    if ctrl_col1.button("🟢  Start Camera", key="cam_start", disabled=st.session_state.cam_running):
        st.session_state.cam_running = True
        st.session_state.cam_tracker = SORT()
        st.session_state.cam_heatmap = None
        st.session_state.cam_graph   = []
        st.session_state.cam_peak    = 0
        st.rerun()

    if ctrl_col2.button("🔴  Stop & Reset", key="cam_stop", disabled=not st.session_state.cam_running):
        st.session_state.cam_running = False
        st.rerun()

    if not st.session_state.cam_running:
        st.markdown("""
        <div style='background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 60px; text-align: center;'>
            <h3 style='color: #94a3b8 !important; margin-bottom: 8px;'>Signal Offline</h3>
            <p style='color: #64748b; font-size: 0.875rem;'>Initiate 'Start Camera' to bridge the live stream</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        if source == "Camera":
            cap = cv2.VideoCapture(0)
        else:
            cctv_url = st.session_state.get("cctv_ip_input", "")
            if not cctv_url:
                st.warning("Please configure your CCTV IP in the sidebar.")
                st.stop()
            cap = cv2.VideoCapture(cctv_url)
        
        last_process_time = 0
        last_db_log_time = 0
        
        while st.session_state.cam_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read camera")
                break
                
            current_time = time.time()
            
            # Process frame and update UI without harsh delays (0.03s limit caps at ~30 FPS)
            if current_time - last_process_time >= 0.03:
                last_process_time = current_time

                if st.session_state.cam_heatmap is None:
                    h, w = frame.shape[:2]
                    st.session_state.cam_heatmap = Heatmap(w, h)

                tracker  = st.session_state.cam_tracker
                heatmap  = st.session_state.cam_heatmap

                proc_f, proc_h, count, total_unique = process_frame(frame, tracker, heatmap, conf_thresh, nms_thresh)

                if count > st.session_state.cam_peak:
                    st.session_state.cam_peak = count
                max_count = st.session_state.cam_peak

                color, label, bg = get_dynamic_status(count, area_capacity)
                occ_pct = (count / area_capacity) * 100

                # Throttle database logging to every 2 seconds
                if current_time - last_db_log_time >= 2.0:
                    insert_crowd_log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), count, area_capacity, label)
                    if "EMERGENCY" in label:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        insert_alert(timestamp, label, f"Overcrowding: {count} people")
                        
                        # Generate Unique Snapshot
                        alert_id = str(uuid.uuid4())
                        snapshot_filename = f"snapshot_{alert_id}.jpg"
                        snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_filename)
                        cv2.imwrite(snapshot_path, proc_f)
                        
                        insert_emergency_snapshot(
                            alert_id=alert_id,
                            timestamp=timestamp,
                            location_id="Camera Zone 1",
                            severity=label,
                            capacity_limit=area_capacity,
                            current_count=count,
                            snapshot_path=snapshot_path,
                            resolution_status="Pending"
                        )
                        
                        # SMS Throttle (5 seconds)
                        if current_time - st.session_state.last_sms_time >= 5.0:
                            target_phone = os.environ.get("TWILIO_DEST_PHONE", "+1234567890")
                            send_emergency_sms_async(
                                emergency_type=f"{label} ({count} people detected)", 
                                location="Camera Zone 1", 
                                to_phone_number=target_phone
                            )
                            
                            if label == "HIGH EMERGENCY":
                                dispatch_cap_alert(alert_id, count, area_capacity, "Camera Zone 1", label)
                                
                            st.session_state.last_sms_time = current_time
                            
                    last_db_log_time = current_time

                status_html = (
                    f'<div class="status-banner" style="border-left-color: {color};">'
                    f'<span style="display: flex; align-items: center; gap: 12px;">'
                    f'<span style="width: 10px; height: 10px; background: {color}; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px {color};"></span>'
                    f'<span style="color: #f8fafc; font-weight: 600;">STATUS: <span style="color: {color};">{label}</span></span>'
                    f'</span>'
                    f'<span style="color: #94a3b8; font-size: 0.875rem;">AI tracking {count} unique signatures</span>'
                    f'</div>'
                )

                st.session_state.cam_graph.append(count)
                cache_ui_state(status_html, count, label, occ_pct, max_count, total_unique, proc_f, proc_h, st.session_state.cam_graph)

                st.session_state.cam_tracker = tracker
                st.session_state.cam_heatmap = heatmap
        
        if cap.isOpened():
            cap.release()

elif source == "Video":
    if video_file:
        # Persistent temp file handler so UI reruns don't erase the currently loaded video
        if "vid_name" not in st.session_state or st.session_state.vid_name != video_file.name:
            if "vid_tfile" in st.session_state and os.path.exists(st.session_state.vid_tfile):
                try: os.unlink(st.session_state.vid_tfile)
                except: pass
            
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(video_file.read())
            tfile.close()
            
            st.session_state.vid_tfile = tfile.name
            st.session_state.vid_name = video_file.name
            st.session_state.vid_frame = 0
            st.session_state.vid_playing = False
            st.session_state.vid_graph = []
            st.session_state.vid_tracker = SORT()

        cap = cv2.VideoCapture(st.session_state.vid_tfile)
        
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(3))
            height = int(cap.get(4))

            # --- Video Playback Controls UI ---
            st.markdown("<h3 style='color: #f8fafc; margin-bottom: 8px; font-size: 1.15rem; font-weight: 600;'>Forensic Playback Controls</h3>", unsafe_allow_html=True)
            
            # Use smaller, tighter columns for professional compact buttons
            control_col1, control_col2, control_col3, control_col4, _ = st.columns([0.8, 0.8, 0.8, 0.8, 6])
            
            with control_col1:
                if st.button("⏪ -5s"):
                    st.session_state.vid_frame = max(0, st.session_state.vid_frame - int(fps * 5))
            with control_col2:
                if st.button("⏸ Pause" if st.session_state.vid_playing else "▶ Play"):
                    st.session_state.vid_playing = not st.session_state.vid_playing
            with control_col3:
                if st.button("⏩ +5s"):
                    st.session_state.vid_frame = min(total_frames - 1, st.session_state.vid_frame + int(fps * 5))
            with control_col4:
                if st.button("⏹ Reset"):
                    st.session_state.vid_playing = False
                    st.session_state.vid_frame = 0
                    st.session_state.vid_graph = []
                    st.session_state.vid_tracker = SORT()

            # Seek Slider
            seek_pos = st.slider("Timeline Tracker", 0, max(1, total_frames - 1), st.session_state.vid_frame, key="vid_seeker")
            if seek_pos != st.session_state.vid_frame:
                st.session_state.vid_frame = seek_pos
                st.session_state.vid_playing = False  # Auto-pause when user grabs slider

            # Speed Control
            playback_speed = st.slider("Playback Speed Multiplier (Virtual FPS)", 1, 10, 3, help="Higher values increase video playback speed by skipping intermediate AI processing frames.")

            # Sync opencv to correct frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.vid_frame)

            heatmap = Heatmap(width, height)
            tracker = st.session_state.vid_tracker

            # --- Playback Logic ---
            if not st.session_state.vid_playing:
                # Render paused static frame
                ret, frame = cap.read()
                if ret:
                    proc_f, proc_h, count, total_unique = process_frame(frame, tracker, heatmap, conf_thresh, nms_thresh)
                    pause_markup = f'<div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); color: #60a5fa; padding: 12px; border-radius: 8px; text-align: center; font-weight: 500; margin-bottom: 24px; font-size: 0.875rem;">⏸ PAUSED at Frame {st.session_state.vid_frame} / {total_frames}</div>'
                    cache_ui_state(pause_markup, count, "PAUSED", (count / area_capacity) * 100, max_count, total_unique, proc_f, proc_h, st.session_state.vid_graph)
            else:
                while cap.isOpened() and st.session_state.vid_playing:
                    loop_start = time.time()
                    ret, frame = cap.read()
                    if not ret: 
                        st.session_state.vid_playing = False
                        st.rerun()

                    st.session_state.vid_frame += 1
                    
                    # Target specific frames for AI processing based on playback speed slider 
                    # All other frames are rendered as normally moving footage to maintain 30FPS smoothness!
                    run_ai = (st.session_state.vid_frame % playback_speed == 0)

                    proc_f, proc_h, count, total_unique = process_frame(frame, tracker, heatmap, conf_thresh, nms_thresh, run_ai=run_ai)
                    if count > max_count: max_count = count

                    color, label, bg = get_dynamic_status(count, area_capacity)
                    occ_pct = (count / area_capacity) * 100

                    current_time = time.time()
                    if "EMERGENCY" in label:
                        if current_time - st.session_state.last_sms_time >= 5.0:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            # Generate Unique Snapshot
                            alert_id = str(uuid.uuid4())
                            snapshot_filename = f"snapshot_{alert_id}.jpg"
                            snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_filename)
                            cv2.imwrite(snapshot_path, proc_f)
                            
                            insert_emergency_snapshot(
                                alert_id=alert_id,
                                timestamp=timestamp,
                                location_id="Video Forensic Analysis",
                                severity=label,
                                capacity_limit=area_capacity,
                                current_count=count,
                                snapshot_path=snapshot_path,
                                resolution_status="Pending"
                            )
                            
                            target_phone = os.environ.get("TWILIO_DEST_PHONE", "+1234567890")
                            send_emergency_sms_async(
                                emergency_type=f"{label} ({count} people detected)", 
                                location="Video Forensic Analysis", 
                                to_phone_number=target_phone
                            )
                            
                            if label == "HIGH EMERGENCY":
                                dispatch_cap_alert(alert_id, count, area_capacity, "Video Forensic Analysis", label)
                                
                            st.session_state.last_sms_time = current_time

                    status_html = (
                        f'<div class="status-banner" style="border-left-color: {color};">'
                        f'<span style="display: flex; align-items: center; gap: 12px;">'
                        f'<span style="width: 10px; height: 10px; background: {color}; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px {color};"></span>'
                        f'<span style="color: #f8fafc; font-weight: 600;">STATUS: <span style="color: {color};">{label}</span></span>'
                        f'</span>'
                        f'<span style="color: #94a3b8; font-size: 0.875rem;">Forensic Analysis: Frame {st.session_state.vid_frame}/{total_frames}</span>'
                        f'</div>'
                    )

                    st.session_state.vid_graph.append(count)
                    cache_ui_state(status_html, count, label, occ_pct, max_count, total_unique, proc_f, proc_h, st.session_state.vid_graph)
                    
                    st.session_state.vid_tracker = tracker

                    # --- Enforce 30 FPS Smoothness (33.3ms per frame) ---
                    # Prevents the UI from blasting erratic frames or tearing while syncing cleanly to Real-Time bounds.
                    elapsed = time.time() - loop_start
                    if elapsed < 0.0333:
                        time.sleep(0.0333 - elapsed)

            cap.release()

elif source == "Image":
    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        heatmap = Heatmap(frame.shape[1], frame.shape[0])
        tracker = SORT(min_hits=1, max_age=3)
        proc_f, proc_h, count, total_unique = process_frame(frame, tracker, heatmap, conf_thresh, nms_thresh)
        color, label, bg = get_dynamic_status(count, area_capacity)
        occ_pct = (count / area_capacity) * 100

        current_time = time.time()
        if "EMERGENCY" in label:
            if current_time - st.session_state.last_sms_time >= 5.0:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Generate Unique Snapshot
                alert_id = str(uuid.uuid4())
                snapshot_filename = f"snapshot_{alert_id}.jpg"
                snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_filename)
                cv2.imwrite(snapshot_path, proc_f)
                
                insert_emergency_snapshot(
                    alert_id=alert_id,
                    timestamp=timestamp,
                    location_id="Static Image Analysis",
                    severity=label,
                    capacity_limit=area_capacity,
                    current_count=count,
                    snapshot_path=snapshot_path,
                    resolution_status="Pending"
                )
                
                target_phone = os.environ.get("TWILIO_DEST_PHONE", "+1234567890")
                send_emergency_sms_async(
                    emergency_type=f"{label} ({count} people detected)", 
                    location="Static Image Analysis", 
                    to_phone_number=target_phone
                )
                
                if label == "HIGH EMERGENCY":
                    dispatch_cap_alert(alert_id, count, area_capacity, "Static Image Analysis", label)
                    
                st.session_state.last_sms_time = current_time

        status_placeholder.markdown(
            f'<div class="status-banner" style="border-left-color: {color};">'
            f'<span style="display: flex; align-items: center; gap: 12px;">'
            f'<span style="width: 10px; height: 10px; background: {color}; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px {color};"></span>'
            f'<span style="color: #f8fafc; font-weight: 600;">STATUS: <span style="color: {color};">{label}</span></span>'
            f'</span>'
            f'<span style="color: #94a3b8; font-size: 0.875rem;">Static Signature Analysis Completed</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        count_metric.metric("Live Count", count)
        status_metric.metric("Safety", label)
        occupancy_metric.metric("Occupancy", f"{int(occ_pct)}%")
        total_metric.metric("Total Seen", total_unique)

        FRAME_WINDOW.image(cv2.cvtColor(proc_f, cv2.COLOR_BGR2RGB))
        HEAT_WINDOW.image(cv2.cvtColor(proc_h, cv2.COLOR_BGR2RGB))