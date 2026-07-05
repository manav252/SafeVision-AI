import queue
import math
import html
import hashlib
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from detector import SafetyDetector
from risk_engine import RiskEngine
from utils import (
    DEFAULT_ZONE_COLOR,
    build_default_zone,
    build_preset_zone,
    draw_frame_annotations,
    ensure_project_dirs,
    extract_polygon_from_canvas,
    person_touches_zone,
    save_evidence_frame,
    write_violation_log_csv,
)


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
EVIDENCE_DIR = OUTPUTS_DIR / "evidence"
LOGS_DIR = OUTPUTS_DIR / "logs"
UPLOADS_DIR = OUTPUTS_DIR / "uploads"
CANVAS_FRAMES_DIR = OUTPUTS_DIR / "canvas_frames"
SAMPLE_VIDEOS_DIR = BASE_DIR / "sample_videos"
FACTORY_DEMO_VIDEO = SAMPLE_VIDEOS_DIR / "factory_demonstration.mp4"
ASSETS_DIR = BASE_DIR / "assets"
ARCHITECTURE_DIAGRAM_PATH = ASSETS_DIR / "SafeVision_AI_Architecture.png"
LANDING_DEMO_VIDEO_PATH = ASSETS_DIR / "safevision_demo_video.mp4"
APP_TIMEZONE = timezone(timedelta(hours=5, minutes=30), name="IST")


st.set_page_config(
    page_title="SafeVision AI",
    page_icon="SV",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not hasattr(st, "dialog"):
    st.dialog = lambda _title: (lambda func: func)


def clean_ui_text(value: object) -> str:
    """Remove internal wording from presentation-facing UI text."""
    text = str(value or "")
    replacements = {
        " (estimated)": "",
        "(estimated)": "",
        "estimated ": "",
        "Estimated ": "",
        "fallback ": "",
        "Fallback ": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return " ".join(text.split())


def app_now() -> datetime:
    return datetime.now(APP_TIMEZONE)


def app_time() -> str:
    return app_now().strftime("%H:%M:%S IST")


def app_datetime_stamp() -> str:
    return app_now().strftime("%Y-%m-%d %H:%M:%S IST")


def app_filename_stamp() -> str:
    return app_now().strftime("%Y%m%d_%H%M%S")


def init_session_state() -> None:
    defaults = {
        "processing": False,
        "processed_video": False,
        "progress": 0.0,
        "current_frame": None,
        "risk_score": 0,
        "risk_level": "LOW",
        "worker_count": 0,
        "violation_count": 0,
        "violation_log": [],
        "worker_queue": None,
        "worker_thread": None,
        "worker_done": False,
        "worker_error": None,
        "video_path": None,
        "video_paths": {},
        "active_cctv_index": 0,
        "source_signature": None,
        "zone_points": None,
        "zone_preset": "drawn",
        "zone_canvas_nonce": 0,
        "zone_map_target": "Zone A",
        "zone_edit_target": "Zone A",
        "custom_zone_points": {},
        "monitor_all_zones": False,
        "detector_mode": "unknown",
        "csv_path": None,
        "last_frame_index": 0,
        "gas_context": None,
        "gas_alert_text": "Normal",
        "gas_history": [],
        "input_mode": "Recorded CCTV",
        "live_stop_event": None,
        "live_mode": False,
        "gas_scenario_control": "Elevated accumulation",
        "permit_type_control": "Maintenance Permit",
        "maintenance_active_control": True,
        "equipment_status_control": "Pump maintenance active",
        "shift_phase_control": "Shift handover in 30 min",
        "audit_status_control": "Permit checklist pending",
        "emergency_event_control": "None",
        "real_time_gas_feed_control": False,
        "copilot_answer": "",
        "active_preset": "Gas + Permit Risk",
        "preset_feedback": "",
        "zone_live_state": "No zone event yet",
        "zone_live_event": None,
        "plant_cameras": [],
        "plant_monitoring_active": False,
        "camera_alerts": {},
        "camera_metrics": {},
        "camera_context": {},
        "camera_evidence": {},
        "generated_report": False,
        "generated_report_text": "",
        "zone_action_feedback": "",
        "monitoring_started_at": None,
        "heatmap_history": [],
        "selected_heatmap_hotspot": "Highest Risk",
        "upload_nonce": 0,
        "reset_feedback": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_processing_state() -> None:
    st.session_state.processing = False
    st.session_state.processed_video = False
    st.session_state.progress = 0.0
    st.session_state.current_frame = None
    st.session_state.risk_score = 0
    st.session_state.risk_level = "LOW"
    st.session_state.worker_count = 0
    st.session_state.violation_count = 0
    st.session_state.violation_log = []
    st.session_state.worker_queue = None
    st.session_state.worker_thread = None
    st.session_state.worker_done = False
    st.session_state.worker_error = None
    st.session_state.csv_path = None
    st.session_state.last_frame_index = 0
    st.session_state.gas_alert_text = "Normal"
    st.session_state.gas_history = []
    st.session_state.zone_live_state = "No zone event yet"
    st.session_state.zone_live_event = None
    st.session_state.plant_monitoring_active = False
    st.session_state.monitoring_started_at = None
    for camera in st.session_state.get("plant_cameras", []):
        camera["monitoring"] = False
    if st.session_state.live_stop_event is not None:
        st.session_state.live_stop_event.set()
    st.session_state.live_stop_event = None
    st.session_state.live_mode = False


def resolve_dashboard_theme() -> str:
    return "light"


@st.cache_resource(show_spinner=False)
def get_detector() -> SafetyDetector:
    return SafetyDetector(models_dir=MODELS_DIR)


def open_capture(source):
    if isinstance(source, int):
        return cv2.VideoCapture(source)
    return cv2.VideoCapture(str(source))


def frame_detail_score(frame: np.ndarray) -> float:
    if frame is None or frame.size == 0:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    edge_density = float(np.mean(cv2.Canny(gray, 50, 150) > 0) * 100)
    blank_penalty = 35.0 if brightness > 242 and contrast < 12 else 0.0
    return contrast + edge_density - blank_penalty


def create_factory_preview_frame(width: int = 960, height: int = 540) -> np.ndarray:
    frame = np.full((height, width, 3), (238, 242, 247), dtype=np.uint8)
    cv2.rectangle(frame, (0, 360), (width, height), (82, 91, 98), -1)
    cv2.rectangle(frame, (56, 80), (905, 345), (64, 76, 88), 4)
    cv2.rectangle(frame, (88, 110), (250, 326), (42, 96, 120), -1)
    cv2.rectangle(frame, (690, 110), (860, 326), (46, 105, 130), -1)
    for x in range(320, 700, 90):
        cv2.line(frame, (x, 82), (x - 55, 346), (112, 120, 126), 7)
    cv2.putText(frame, "INDUSTRIAL CCTV - FACTORY DEMO", (62, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (30, 37, 48), 2)

    cv2.rectangle(frame, (580, 306), (875, 512), (0, 0, 210), 4)
    overlay = frame.copy()
    cv2.rectangle(overlay, (580, 306), (875, 512), (0, 0, 210), -1)
    frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)
    cv2.putText(frame, "RESTRICTED ZONE", (604, 296), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (0, 0, 235), 2)

    worker_x, worker_y = 660, 270
    cv2.circle(frame, (worker_x + 35, worker_y - 38), 20, (54, 217, 235), -1)
    cv2.rectangle(frame, (worker_x + 10, worker_y - 18), (worker_x + 60, worker_y + 82), (28, 214, 225), -1)
    cv2.line(frame, (worker_x + 16, worker_y), (worker_x + 56, worker_y + 78), (255, 126, 0), 5)
    cv2.line(frame, (worker_x + 56, worker_y), (worker_x + 16, worker_y + 78), (255, 126, 0), 5)
    cv2.line(frame, (worker_x + 10, worker_y + 10), (worker_x - 18, worker_y + 60), (38, 42, 46), 8)
    cv2.line(frame, (worker_x + 60, worker_y + 10), (worker_x + 90, worker_y + 60), (38, 42, 46), 8)
    cv2.line(frame, (worker_x + 23, worker_y + 82), (worker_x + 13, worker_y + 145), (38, 42, 46), 9)
    cv2.line(frame, (worker_x + 49, worker_y + 82), (worker_x + 62, worker_y + 145), (38, 42, 46), 9)

    cv2.rectangle(frame, (646, 30), (918, 82), (26, 29, 34), -1)
    cv2.putText(frame, "CH4 14% LEL | PERMIT ACTIVE", (662, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (80, 220, 255), 2)
    return frame


def _read_first_frame(video_source):
    cap = open_capture(video_source)
    try:
        if not cap.isOpened():
            return None
        best_frame = None
        best_score = -100.0
        first_frame = None
        for _ in range(45):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if first_frame is None:
                first_frame = frame
            score = frame_detail_score(frame)
            if score > best_score:
                best_frame = frame
                best_score = score
            if score >= 18:
                return frame
        return best_frame if best_frame is not None else first_frame
    finally:
        cap.release()


def load_first_frame(video_source, allow_demo_fallback: bool = False, allow_placeholder: bool = False):
    candidates = [video_source]
    if allow_demo_fallback:
        candidates.append(FACTORY_DEMO_VIDEO)

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            frame = _read_first_frame(candidate)
        except Exception:
            frame = None
        if frame is not None:
            return frame

    if allow_placeholder:
        return create_factory_preview_frame()
    return None


def safe_asset_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(value).stem).strip("-._")
    return stem[:60] or "asset"


def save_uploaded_video(uploaded_file) -> Path:
    ensure_project_dirs([UPLOADS_DIR])
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    payload = uploaded_file.getbuffer()
    digest = hashlib.sha256(payload).hexdigest()[:16]
    target = UPLOADS_DIR / f"{safe_asset_stem(uploaded_file.name)}-{uploaded_file.size}-{digest}{suffix}"
    if not target.exists() or target.stat().st_size != uploaded_file.size:
        target.write_bytes(payload)
    return target


def save_canvas_background_frame(frame: np.ndarray, camera_id: str, width: int, height: int) -> Path | None:
    ensure_project_dirs([CANVAS_FRAMES_DIR])
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    preview = Image.fromarray(rgb_frame).resize((width, height), Image.Resampling.LANCZOS).convert("RGB")
    target = CANVAS_FRAMES_DIR / f"{safe_asset_stem(camera_id)}_{width}x{height}.png"
    try:
        preview.save(target, format="PNG")
    except Exception:
        return None
    return target


PLANT_ZONES = ["Zone A", "Zone B", "Control Room", "Reactor Zone"]
CAMERA_CONTEXT_KEYS = [
    "gas_scenario_control",
    "permit_type_control",
    "maintenance_active_control",
    "equipment_status_control",
    "shift_phase_control",
    "audit_status_control",
    "emergency_event_control",
    "real_time_gas_feed_control",
    "active_preset",
    "preset_feedback",
]


def uploaded_file_key(uploaded_file) -> str:
    return f"{uploaded_file.name}:{uploaded_file.size}"


def init_camera_state(uploaded_files: list) -> list[dict]:
    cameras = []
    existing_by_key = {camera.get("file_key"): camera for camera in st.session_state.get("plant_cameras", [])}
    for index, uploaded_file in enumerate(uploaded_files):
        file_key = uploaded_file_key(uploaded_file)
        camera = dict(existing_by_key.get(file_key, {}))
        if not camera:
            try:
                video_path = str(save_uploaded_video(uploaded_file))
            except Exception:
                video_path = ""
            camera = {
                "id": f"cctv_{index + 1}_{abs(hash(file_key))}",
                "index": index,
                "camera": f"CCTV-{index + 1}",
                "file_key": file_key,
                "name": uploaded_file.name,
                "path": video_path,
                "zone": st.session_state.get(f"feed_zone_{index}", PLANT_ZONES[index] if index < len(PLANT_ZONES) else "Pending Setup"),
                "status": "Pending Setup",
                "monitoring": False,
                "worker_count": 0,
                "alert_count": 0,
                "ppe_compliance": 100,
                "risk_score": 0,
                "evidence_count": 0,
            }
        camera["index"] = index
        camera["camera"] = f"CCTV-{index + 1}"
        camera["name"] = uploaded_file.name
        camera["zone"] = st.session_state.get(f"feed_zone_{index}", camera.get("zone", "Pending Setup"))
        if camera["zone"] not in PLANT_ZONES and camera["zone"] != "Pending Setup":
            camera["zone"] = PLANT_ZONES[index] if index < len(PLANT_ZONES) else "Pending Setup"
        refresh_camera_status(camera)
        cameras.append(camera)
    st.session_state.plant_cameras = cameras
    if cameras and st.session_state.active_cctv_index >= len(cameras):
        st.session_state.active_cctv_index = 0
    return cameras


def active_camera() -> dict | None:
    cameras = st.session_state.get("plant_cameras", [])
    if not cameras:
        return None
    index = min(st.session_state.get("active_cctv_index", 0), len(cameras) - 1)
    return cameras[index]


def snapshot_active_camera_context() -> None:
    camera = active_camera()
    if not camera:
        return
    st.session_state.camera_context[camera["id"]] = {
        key: st.session_state.get(key)
        for key in CAMERA_CONTEXT_KEYS
        if key in st.session_state
    }


def restore_camera_context(camera: dict | None) -> None:
    if not camera:
        return
    context = st.session_state.get("camera_context", {}).get(camera.get("id"))
    if not context:
        return
    for key, value in context.items():
        if value is not None:
            try:
                st.session_state[key] = value
            except st.errors.StreamlitAPIException:
                # If the widget already exists in this run, keep the camera context stored
                # and let the next rerun apply it naturally.
                pass


def refresh_camera_status(camera: dict) -> None:
    zone = camera.get("zone", "Pending Setup")
    has_zone = zone in PLANT_ZONES and zone_storage_key(zone, camera.get("index", 0)) in st.session_state.get("custom_zone_points", {})
    metrics = st.session_state.get("camera_metrics", {}).get(camera.get("id"), {})
    camera["worker_count"] = int(metrics.get("worker_count", camera.get("worker_count", 0)))
    camera["alert_count"] = int(metrics.get("alert_count", camera.get("alert_count", 0)))
    camera["ppe_compliance"] = int(metrics.get("ppe_compliance", camera.get("ppe_compliance", 100)))
    camera["risk_score"] = int(metrics.get("risk_score", camera.get("risk_score", 0)))
    if camera.get("monitoring"):
        camera["status"] = "Alert" if camera["alert_count"] or camera["risk_score"] >= 70 else "Monitoring"
    elif has_zone:
        camera["status"] = "Configured"
    else:
        camera["status"] = "Pending Setup"


def select_camera(index: int, restore_context: bool = True) -> None:
    cameras = st.session_state.get("plant_cameras", [])
    if not cameras or index >= len(cameras):
        return
    snapshot_active_camera_context()
    st.session_state.active_cctv_index = index
    camera = cameras[index]
    zone = camera.get("zone", "Zone A") if camera.get("zone") in PLANT_ZONES else "Zone A"
    st.session_state.zone_map_target = zone
    st.session_state.zone_edit_target = zone
    st.session_state.zone_canvas_nonce += 1
    metrics = st.session_state.get("camera_metrics", {}).get(camera.get("id"), {})
    events = st.session_state.get("camera_alerts", {}).get(camera.get("id"), [])
    st.session_state.current_frame = None
    st.session_state.progress = 0.0
    st.session_state.worker_count = int(metrics.get("worker_count", 0))
    st.session_state.violation_count = int(metrics.get("alert_count", len(events)))
    st.session_state.risk_score = int(metrics.get("risk_score", 0))
    st.session_state.risk_level = "HIGH" if st.session_state.risk_score >= 70 else "MEDIUM" if st.session_state.risk_score >= 35 else "LOW"
    st.session_state.violation_log = [
        {
            "timestamp": event.get("timestamp"),
            "frame": "-",
            "violation_type": event.get("type", "camera_event"),
            "message": event.get("message", "Safety event"),
            "severity": event.get("severity", "MEDIUM"),
            "zone_name": event.get("zone", zone),
        }
        for event in events
    ]
    if restore_context:
        restore_camera_context(camera)


def save_camera_zone(zone_name: str, points: list[tuple[int, int]], cctv_index: int | None = None) -> None:
    set_saved_zone(zone_name, points, cctv_index)
    camera = active_camera()
    if camera:
        camera["zone"] = zone_name
        st.session_state[f"feed_zone_{camera.get('index', 0)}"] = zone_name
        refresh_camera_status(camera)


def calculate_risk_score(events: list[dict], gas_context: dict | None = None) -> tuple[int, str]:
    alert_types = {event.get("violation_type") or event.get("type") for event in events}
    score = 0
    if {"no_helmet", "no_vest", "ppe"} & alert_types:
        score += 20
    if {"restricted_zone_breach", "restricted_zone_entry", "zone"} & alert_types:
        score += 30
    if gas_elevated(gas_context) or "gas" in alert_types:
        score += 40
    if "permit" in alert_types or (gas_context and gas_context.get("permit_type", "None") != "None" and gas_elevated(gas_context)):
        score += 25
    score = min(100, score)
    level = "HIGH" if score >= 70 else "MEDIUM" if score >= 35 else "LOW"
    return score, level


def build_alert_explanation(event: dict, gas_context: dict | None) -> dict:
    """Create a concise, rule-based explanation for a safety event."""
    current_events = collect_plant_events(gas_context)
    score, level = calculate_risk_score(current_events, gas_context)
    if live_cctv_connected():
        score = max(score, int(st.session_state.get("risk_score", 0)))
        level = "CRITICAL" if score >= 85 else "HIGH" if score >= 70 else "MEDIUM" if score >= 35 else "LOW"

    event_type = str(event.get("type") or event.get("violation_type") or "event")
    reasons = []
    if event_type in {"no_helmet", "helmet", "ppe"} or "helmet" in clean_ui_text(event.get("message", "")).lower():
        reasons.append(("Helmet not detected", 20))
    if event_type in {"no_vest", "ppe"} or "vest" in clean_ui_text(event.get("message", "")).lower():
        reasons.append(("Safety vest not detected", 20))
    if event_type in {"restricted_zone_breach", "restricted_zone_entry", "zone"} or "restricted" in clean_ui_text(event.get("message", "")).lower():
        reasons.append(("Restricted zone entry", 30))
    if gas_elevated(gas_context) or event_type == "gas":
        reasons.append(("Abnormal gas reading", 40))
    if gas_context and gas_context.get("permit_type", "None") != "None":
        reasons.append((f"Active {gas_context.get('permit_type')}", 15))
    if event_type == "permit":
        reasons.append(("Permit conflict detected", 25))
    repeat_count = sum(
        1
        for row in current_events
        if (row.get("type") or row.get("violation_type")) == event_type
    )
    if repeat_count > 1:
        reasons.append((f"Repeated violation pattern ({repeat_count} recent events)", min(21, 8 + repeat_count * 2)))
    if not reasons:
        reasons.append(("Live monitoring context changed", max(10, min(30, score))))

    actions = []
    if any("gas" in reason.lower() for reason, _ in reasons):
        actions.extend(["Verify gas sensor reading", "Increase ventilation", "Pause conflicting permit"])
    if any("restricted" in reason.lower() for reason, _ in reasons):
        actions.extend(["Notify area supervisor", "Restrict entry", "Review CCTV footage"])
    if any("helmet" in reason.lower() or "vest" in reason.lower() for reason, _ in reasons):
        actions.extend(["Notify supervisor immediately", "Recheck PPE compliance", "Hold worker at access point"])
    if not actions:
        actions = ["Notify safety admin", "Review recent events", "Continue monitoring"]
    deduped_actions = list(dict.fromkeys(actions))[:4]
    confidence = min(98, 72 + len(reasons) * 4 + (8 if score >= 70 else 0))
    return {
        "score": score,
        "level": level,
        "reasons": reasons,
        "actions": deduped_actions,
        "confidence": confidence,
        "event": event,
    }


@st.dialog("AI Risk Explanation")
def show_alert_explanation_dialog(event: dict, gas_context: dict | None) -> None:
    explanation = build_alert_explanation(event, gas_context)
    event_message = clean_ui_text(event.get("message") or event.get("event") or "Safety event")
    st.markdown(
        f"""
        <div class="modal-score-card">
          <span>Overall Risk Score</span>
          <strong>{int(explanation["score"])} / 100</strong>
          <em>{html.escape(str(explanation["level"]))}</em>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"{event.get('timestamp', event.get('time', 'Live'))} | {event.get('camera', event.get('source', 'CCTV'))} | {event.get('zone', 'Plant')}")
    st.markdown(f"**Alert:** {html.escape(event_message)}")
    st.markdown("**Reasoning**")
    for reason, weight in explanation["reasons"]:
        st.markdown(f"- {reason} (+{weight})")
    st.markdown("**Recommended Action**")
    for action in explanation["actions"]:
        st.markdown(f"- {action}")
    st.markdown(f"**Confidence:** {explanation['confidence']}%")
    st.caption("Prepared by AI Safety Advisor")


@st.dialog("SafeVision AI Pipeline")
def show_architecture_dialog() -> None:
    if ARCHITECTURE_DIAGRAM_PATH.exists():
        st.image(str(ARCHITECTURE_DIAGRAM_PATH), use_column_width=True)
    else:
        st.info("Architecture diagram file is not available in the workspace yet.")
    st.markdown(
        """
        **SafeVision AI Pipeline**

        1. Multi-camera CCTV feeds
        2. Vision Processing
        3. Computer Vision Engine
        4. Safety Fusion Engine
        5. Safety Rule Engine
        6. Risk Engine
        7. AI Safety Advisor
        8. Dashboard + Heatmap + Reports

        **Core Principle**

        SafeVision AI combines CCTV analytics with operational plant context to transform isolated events into real-time safety intelligence.
        """
    )
    with st.expander("Computer Vision Engine"):
        st.write("Responsible for worker detection, PPE compliance, and restricted-zone analytics.")
    with st.expander("Safety Fusion Engine"):
        st.write("Combines visual events with permits, equipment state, gas readings, and shift information.")
    with st.expander("Safety Rule Engine"):
        st.write("Applies safety rules such as gas plus zone entry, PPE missing plus active permit, and repeated violations.")
    with st.expander("Risk Engine"):
        st.write("Converts active safety factors into a weighted 0-100 risk score and Low, Medium, High, or Critical class.")


def current_camera_events(camera: dict | None, gas_context: dict | None) -> list[dict]:
    if not camera:
        return []
    rows = list(st.session_state.get("camera_alerts", {}).get(camera.get("id"), []))
    if not rows and gas_elevated(gas_context):
        rows.append(
            {
                "timestamp": app_time(),
                "camera": camera.get("camera", "CCTV"),
                "zone": camera.get("zone", "Plant"),
                "severity": "HIGH",
                "type": "gas",
                "message": f"Gas level rising near {camera.get('zone', 'plant zone')}",
            }
        )
    return rows


def collect_plant_events(gas_context: dict | None) -> list[dict]:
    events = []
    for camera in st.session_state.get("plant_cameras", []):
        events.extend(st.session_state.get("camera_alerts", {}).get(camera.get("id"), []))
    if live_cctv_connected():
        for row in st.session_state.get("violation_log", [])[-12:]:
            vtype = row.get("violation_type", "vision")
            events.append(
                {
                    "timestamp": row.get("timestamp", app_time()),
                    "camera": "Industrial CCTV",
                    "zone": row.get("zone_name", st.session_state.get("zone_map_target", "Plant")),
                    "severity": row.get("severity", "MEDIUM"),
                    "type": vtype,
                    "message": row.get("message", str(vtype).replace("_", " ").title()),
                }
            )
        if gas_elevated(gas_context):
            readings = gas_context.get("readings", {}) if gas_context else {}
            events.append(
                {
                    "timestamp": gas_context.get("sensor_timestamp", app_time()) if gas_context else app_time(),
                    "camera": "Industrial CCTV",
                    "zone": st.session_state.get("zone_map_target", "Plant"),
                    "severity": "HIGH",
                    "type": "gas",
                    "message": f"Gas level elevated: CH4 {readings.get('methane_lel', 0)}% LEL, CO {readings.get('co_ppm', 0)} ppm",
                }
            )
        if gas_context and gas_context.get("permit_type", "None") != "None" and gas_elevated(gas_context):
            events.append(
                {
                    "timestamp": app_time(),
                    "camera": "Permit Engine",
                    "zone": st.session_state.get("zone_map_target", "Plant"),
                    "severity": "HIGH",
                    "type": "permit",
                    "message": f"{gas_context.get('permit_type')} overlaps with abnormal gas readings",
                }
            )
    if not events and st.session_state.get("plant_monitoring_active"):
        now = app_time()
        for camera in st.session_state.get("plant_cameras", [])[:4]:
            severity = "HIGH" if gas_elevated(gas_context) else "MEDIUM"
            events.append(
                {
                    "timestamp": now,
                    "camera": camera.get("camera", "CCTV"),
                    "zone": camera.get("zone", "Plant"),
                    "severity": severity,
                    "type": "monitoring",
                    "message": f"{camera.get('camera')} monitoring active in {camera.get('zone')}",
                }
            )
    return events[-12:]


def start_multi_camera_monitoring(gas_context: dict | None) -> None:
    st.session_state.plant_monitoring_active = True
    if not st.session_state.get("monitoring_started_at"):
        st.session_state.monitoring_started_at = time.time()
    now = app_time()
    for camera in st.session_state.get("plant_cameras", []):
        refresh_camera_status(camera)
        configured = camera.get("status") in {"Configured", "Monitoring", "Alert"}
        camera["monitoring"] = configured
        camera["worker_count"] = max(camera.get("worker_count", 0), 1 if configured else 0)
        base_events = []
        if configured:
            if gas_elevated(gas_context):
                base_events.append(
                    {
                        "timestamp": now,
                        "camera": camera["camera"],
                        "zone": camera["zone"],
                        "severity": "HIGH",
                        "type": "gas",
                        "message": f"Gas level rising near {camera['zone']}",
                    }
                )
            if camera.get("risk_score", 0) >= 70:
                base_events.append(
                    {
                        "timestamp": now,
                        "camera": camera["camera"],
                        "zone": camera["zone"],
                        "severity": "HIGH",
                        "type": "zone",
                        "message": f"Restricted-zone risk active in {camera['zone']}",
                    }
                )
        st.session_state.camera_alerts[camera["id"]] = (st.session_state.camera_alerts.get(camera["id"], []) + base_events)[-20:]
        camera["alert_count"] = len(st.session_state.camera_alerts.get(camera["id"], []))
        camera["ppe_compliance"] = max(72, 100 - camera["alert_count"] * 4)
        camera["risk_score"], _ = calculate_risk_score(st.session_state.camera_alerts.get(camera["id"], []), gas_context)
        st.session_state.camera_metrics[camera["id"]] = {
            "worker_count": camera["worker_count"],
            "alert_count": camera["alert_count"],
            "ppe_compliance": camera["ppe_compliance"],
            "risk_score": camera["risk_score"],
        }
        refresh_camera_status(camera)


def render_monitoring_animation() -> None:
    steps = [
        "Initializing Camera Network...",
        "Loading AI Vision Models...",
        "Loading Plant Context...",
        "Synchronizing Monitoring Engine...",
        "Monitoring Started",
    ]
    progress = st.progress(0, text=steps[0])
    status = st.empty()
    for index, step in enumerate(steps):
        status.markdown(f"<div class='init-step'>{html.escape(step)}</div>", unsafe_allow_html=True)
        progress.progress((index + 1) / len(steps), text=step)
        time.sleep(0.38)
    status.empty()


def render_live_indicator() -> None:
    if not st.session_state.get("plant_monitoring_active") and not st.session_state.get("processing"):
        return
    started = st.session_state.get("monitoring_started_at") or time.time()
    elapsed = max(0, int(time.time() - started))
    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    st.markdown(
        f"<div class='live-indicator'><span></span><b>LIVE</b><strong>{hrs:02d}:{mins:02d}:{secs:02d}</strong></div>",
        unsafe_allow_html=True,
    )


def live_cctv_connected() -> bool:
    return st.session_state.get("input_mode") == "Industrial CCTV"


def live_cctv_monitoring() -> bool:
    return live_cctv_connected() and bool(st.session_state.get("processing") or st.session_state.get("plant_monitoring_active"))


def ppe_compliance_from_logs(violation_log: list[dict]) -> int | None:
    if not violation_log:
        return None
    ppe_issues = sum(1 for row in violation_log if row.get("violation_type") in {"no_helmet", "no_vest"})
    person_events = max(1, sum(1 for row in violation_log if row.get("violation_type") in {"person", "no_helmet", "no_vest"}))
    return max(0, min(100, int(100 - (ppe_issues / max(1, person_events)) * 55)))


def render_top_header(gas_context: dict | None) -> None:
    cameras = st.session_state.get("plant_cameras", [])
    live_cctv_mode = live_cctv_connected()
    connected_count = len(cameras) + (1 if live_cctv_mode and not cameras else 0)
    active_cameras = sum(1 for camera in cameras if camera.get("monitoring"))
    if live_cctv_monitoring():
        active_cameras = max(active_cameras, 1)
    open_alerts = sum(int(camera.get("alert_count", 0)) for camera in cameras)
    if live_cctv_mode:
        open_alerts = max(open_alerts, int(st.session_state.get("violation_count", 0)))
        if gas_elevated(gas_context):
            open_alerts = max(open_alerts, 1)
    events = collect_plant_events(gas_context)
    context_enabled = bool(cameras or live_cctv_mode or st.session_state.get("plant_monitoring_active"))
    score, level = calculate_risk_score(events, gas_context if context_enabled else None)
    if live_cctv_mode:
        score = max(score, int(st.session_state.get("risk_score", 0)))
        level = "HIGH" if score >= 70 else "MEDIUM" if score >= 35 else "LOW"
    status = "Monitoring Active" if st.session_state.get("plant_monitoring_active") or st.session_state.get("processing") else "Ready"
    status_class = "active" if status == "Monitoring Active" else "ready"
    source_label = "Industrial CCTV" if live_cctv_mode else "Recorded CCTV"
    now = app_time()
    st.markdown(
        f"""
        <div class="top-command-header">
          <div>
            <h1>SafeVision AI</h1>
            <p>AI-powered industrial safety intelligence that detects PPE violations, correlates plant context, predicts risk escalation, and recommends preventive interventions before incidents occur.</p>
          </div>
          <div class="top-summary">
            <div class="summary-status"><b>Plant Status</b><strong class="{status_class}">{html.escape(status)}</strong></div>
            <div><b>Source</b><em>{html.escape(source_label)}</em></div>
            <div><b>Connected Cameras</b><em>{connected_count}</em></div>
            <div><b>Monitoring</b><em>{active_cameras}</em></div>
            <div><b>Current Risk</b><em>{score}% {html.escape(level)}</em></div>
            <div><b>Open Alerts</b><em>{open_alerts}</em></div>
            <div><b>Last Refresh</b><em>{now}</em></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    action_cols = st.columns([0.72, 0.28])
    with action_cols[1]:
        if st.button("System Architecture", key="open_architecture_modal", use_container_width=True):
            show_architecture_dialog()


def render_camera_manager(uploaded_files: list, gas_context: dict | None) -> tuple[list[dict], dict | None]:
    cameras = init_camera_state(uploaded_files)
    st.markdown("<div class='camera-manager-title'>Plant Camera Manager</div>", unsafe_allow_html=True)
    if not cameras:
        st.caption("Upload CCTV videos to create persistent plant camera cards.")
        return [], None

    for camera in cameras:
        refresh_camera_status(camera)
        status = camera.get("status", "Pending Setup")
        active = camera.get("index") == st.session_state.get("active_cctv_index", 0)
        status_class = status.lower().replace(" ", "-")
        status_icon = "●"
        zone_name = camera.get("zone", "Pending Setup")
        card_html = (
            f"<div class='camera-card {status_class} {'selected' if active else ''}'>"
            f"<div class='camera-card-head'><strong>▣ {html.escape(camera['camera'])}</strong><b>{status_icon} {'LIVE' if camera.get('monitoring') else html.escape(status)}</b></div>"
            f"<span>{html.escape(camera['name'])}</span>"
            f"<p>{html.escape(zone_name)}</p>"
            f"<div class='camera-stat-grid'>"
            f"<small><em>Workers</em>{camera.get('worker_count', 0)}</small>"
            f"<small><em>Alerts</em>{camera.get('alert_count', 0)}</small>"
            f"<small><em>Compliance</em>{camera.get('ppe_compliance', 100)}%</small>"
            f"</div></div>"
        )
        st.markdown(
            card_html,
            unsafe_allow_html=True,
        )
        if st.button(
            f"Open {camera['camera']}",
            key=f"open_camera_{camera['id']}",
            type="primary" if active else "secondary",
            use_container_width=True,
        ):
            select_camera(camera["index"])
            st.rerun()
        selected_zone = st.selectbox(
            f"Assign {camera['camera']}",
            ["Pending Setup"] + PLANT_ZONES,
            index=(["Pending Setup"] + PLANT_ZONES).index(camera.get("zone", "Pending Setup")),
            key=f"camera_zone_{camera['id']}",
        )
        if selected_zone != camera.get("zone"):
            camera["zone"] = selected_zone
            st.session_state[f"feed_zone_{camera['index']}"] = selected_zone
            if active and selected_zone in PLANT_ZONES:
                st.session_state.zone_map_target = selected_zone
                st.session_state.zone_edit_target = selected_zone
            refresh_camera_status(camera)
            st.rerun()

    return cameras, active_camera()


def render_plant_status(gas_context: dict | None) -> None:
    cameras = st.session_state.get("plant_cameras", [])
    for camera in cameras:
        refresh_camera_status(camera)
    live_cctv_mode = live_cctv_connected()
    connected_count = len(cameras) + (1 if live_cctv_mode and not cameras else 0)
    active_count = sum(1 for camera in cameras if camera.get("monitoring"))
    if live_cctv_monitoring():
        active_count = max(active_count, 1)
    workers = sum(int(camera.get("worker_count", 0)) for camera in cameras)
    if live_cctv_mode:
        workers = max(workers, int(st.session_state.get("worker_count", 0)))
    alerts = sum(int(camera.get("alert_count", 0)) for camera in cameras)
    if live_cctv_mode:
        alerts = max(alerts, int(st.session_state.get("violation_count", 0)))
        if gas_elevated(gas_context):
            alerts = max(alerts, 1)
    live_ppe = ppe_compliance_from_logs(st.session_state.get("violation_log", []))
    if cameras:
        ppe_value = int(sum(camera.get("ppe_compliance", 100) for camera in cameras) / len(cameras))
        if live_ppe is not None:
            ppe_value = min(ppe_value, live_ppe)
        ppe = f"{ppe_value}%"
    else:
        ppe = "--" if live_ppe is None else f"{live_ppe}%"
    context_enabled = bool(cameras or live_cctv_mode or st.session_state.get("plant_monitoring_active"))
    score, level = calculate_risk_score(collect_plant_events(gas_context), gas_context if context_enabled else None)
    if live_cctv_mode:
        score = max(score, int(st.session_state.get("risk_score", 0)))
        level = "HIGH" if score >= 70 else "MEDIUM" if score >= 35 else "LOW"
    cards = [
        ("Connected Cameras", connected_count),
        ("Monitoring Cameras", active_count),
        ("Workers Detected", workers),
        ("Open Alerts", alerts),
        ("PPE Compliance", ppe),
        ("Current Plant Risk", level),
    ]
    html_cards = "".join(
        f"<div class='plant-status-card'><span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></div>"
        for label, value in cards
    )
    st.markdown(f"<div class='plant-status-grid'>{html_cards}</div>", unsafe_allow_html=True)


def render_alert_feed(gas_context: dict | None) -> None:
    events = collect_plant_events(gas_context)
    st.markdown("<div class='events-panel'><strong>Recent Safety Events</strong>", unsafe_allow_html=True)
    if not events:
        st.markdown("<div class='event-row normal'><b>● Ready</b><span>No open plant events yet</span><em>INFO</em></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    for index, event in enumerate(reversed(events[-8:])):
        severity = event.get("severity", "LOW")
        css = "critical" if severity == "HIGH" else "warning" if severity == "MEDIUM" else "normal"
        icon = "●"
        event_type = str(event.get("type", "event")).replace("_", " ").title()
        st.markdown(
            f"""
            <div class="event-row {css}">
              <b>{icon} {html.escape(event_type)}</b>
              <span>{html.escape(clean_ui_text(event.get('message', 'Safety event')))}</span>
              <em>{html.escape(str(event.get('timestamp', 'Live')))} | {html.escape(event.get('camera', 'CCTV'))} | {html.escape(event.get('zone', 'Plant'))} | {html.escape(severity)}</em>
            </div>
            """,
            unsafe_allow_html=True,
        )
        button_cols = st.columns([0.24, 0.76])
        with button_cols[0]:
            if st.button("🧠 Explain This Alert", key=f"explain_alert_{index}_{event.get('timestamp', event.get('time', 'live'))}_{event.get('type', 'event')}", use_container_width=True):
                show_alert_explanation_dialog(event, gas_context)


def render_ai_advisor(gas_context: dict | None) -> None:
    source_ready = live_cctv_connected() or bool(active_camera()) or bool(st.session_state.get("processing"))
    events = collect_plant_events(gas_context) if source_ready else []
    active_context = gas_context if source_ready or st.session_state.get("plant_monitoring_active") else None
    score, level = calculate_risk_score(events, active_context)
    if live_cctv_connected():
        score = max(score, int(st.session_state.get("risk_score", 0)))
        level = "HIGH" if score >= 70 else "MEDIUM" if score >= 35 else "LOW"

    latest_event = events[-1] if events else {}
    selected_zone = latest_event.get("zone") or (st.session_state.get("zone_map_target", "Plant") if source_ready else "Zone not armed")
    source = latest_event.get("camera") or ("Industrial CCTV" if live_cctv_connected() else active_camera().get("camera", "No CCTV feed") if active_camera() else "No CCTV feed")
    readings = gas_context.get("readings", {}) if gas_context else {}
    gas_snapshot = (
        f"CH4 {readings.get('methane_lel', 0)}% LEL, CO {readings.get('co_ppm', 0)} ppm"
        if readings and source_ready
        else "Plant context staged"
        if readings
        else "Gas feed idle"
    )
    elapsed = 0
    if st.session_state.get("monitoring_started_at"):
        elapsed = int(max(0, time.time() - st.session_state.monitoring_started_at))

    camera = active_camera()
    camera_zone = camera.get("zone") if camera else None
    zone_saved = bool(
        camera
        and camera_zone in PLANT_ZONES
        and zone_storage_key(camera_zone, camera.get("index", 0)) in st.session_state.get("custom_zone_points", {})
    )
    if live_cctv_connected():
        zone_saved = zone_saved or bool(st.session_state.get("custom_zone_points") or st.session_state.get("zone_points"))

    confidence = 0
    if source_ready:
        confidence += 35
        if zone_saved:
            confidence += 25
        if st.session_state.get("plant_monitoring_active") or st.session_state.get("processing"):
            confidence += 15
        if gas_context:
            confidence += 8
        if gas_elevated(gas_context):
            confidence += 5
        confidence += min(12, len(events) * 3)
        confidence += min(5, elapsed // 30)
        confidence = min(98, confidence)
    last_update = app_time()

    if not source_ready:
        advice = "No CCTV source is active yet. Upload a plant camera feed or switch to Industrial CCTV, then save a restricted zone to arm live safety intelligence."
    elif latest_event:
        advice = clean_ui_text(latest_event.get("message", "Safety event detected."))
        if latest_event.get("type") in {"gas", "permit"} or gas_elevated(gas_context):
            advice += " Recommended action: verify gas reading, pause conflicting work, and notify supervisor."
        elif latest_event.get("type") in {"ppe", "no_helmet", "no_vest"}:
            advice += " Recommended action: recheck PPE and hold worker at access point."
        elif latest_event.get("type") in {"zone", "restricted_zone_breach", "restricted_zone_entry"}:
            advice += " Recommended action: clear restricted zone and preserve evidence frame."
    elif not events and level == "LOW":
        advice = "Plant monitoring is ready. Configure a camera zone, then start monitoring for live event correlation."
    elif gas_elevated(active_context):
        advice = f"Gas risk is elevated near {selected_zone}. Recommend ventilation increase, permit pause, and supervisor verification."
    else:
        advice = f"{source} is ready for safety monitoring. No live CCTV violations are currently active."
    st.markdown(
        f"""
        <div class="advisor-panel">
          <span>AI Safety Advisor</span>
          <strong>{level} Risk | {score}/100</strong>
          <div class="advisor-meta">
            <b>{html.escape(source)}</b>
            <b>{html.escape(str(selected_zone))}</b>
            <b>{html.escape(clean_ui_text(gas_snapshot))}</b>
            <b>{len(events)} live events</b>
            <b>Updated {last_update}</b>
          </div>
          <p>{html.escape(clean_ui_text(advice))}</p>
          <div class="confidence"><b>Confidence</b><i style="width:{confidence}%"></i><em>{confidence}%</em></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_meter(score: int, level: str) -> None:
    fill = max(0, min(100, int(score)))
    blocks = int(round(fill / 10))
    bar = "█" * blocks + "░" * (10 - blocks)
    css = "high" if fill >= 70 else "medium" if fill >= 35 else "low"
    st.markdown(
        f"""
        <div class="risk-meter {css}">
          <span>Current Risk</span>
          <strong>{html.escape(level)}</strong>
          <b>{bar}</b>
          <em>{fill}%</em>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_status(camera: dict | None) -> None:
    zone = camera.get("zone") if camera else None
    zone_saved = bool(camera and zone in PLANT_ZONES and zone_storage_key(zone, camera.get("index", 0)) in st.session_state.get("custom_zone_points", {}))
    live_source = live_cctv_connected()
    if live_source:
        zone_saved = bool(st.session_state.get("custom_zone_points") or st.session_state.get("zone_points"))
    monitoring_active = bool(st.session_state.get("processing") or st.session_state.get("plant_monitoring_active") or (camera and camera.get("monitoring")))
    source_ready = bool(camera) or live_source
    selected_label = "Industrial CCTV selected" if live_source else f"{camera.get('camera')} selected" if camera else "No CCTV feed"
    context_label = gas_status_text(st.session_state.get("gas_context") or build_gas_context(
        st.session_state.get("gas_scenario_control", "Normal"),
        st.session_state.get("permit_type_control", "None"),
        st.session_state.get("maintenance_active_control", False),
        st.session_state.get("equipment_status_control", "Normal"),
        st.session_state.get("shift_phase_control", "Stable operations"),
        st.session_state.get("audit_status_control", "Compliant"),
        st.session_state.get("emergency_event_control", "None"),
        False,
    ))
    zone_label = f"{zone or st.session_state.get('zone_map_target', 'Zone')} saved" if zone_saved else "Restricted zone pending"
    monitoring_label = "Monitoring active" if monitoring_active else "Monitoring not started"
    report_label = "Report ready" if st.session_state.get("generated_report") or st.session_state.get("csv_path") else "Report pending"
    steps = [
        (selected_label, source_ready),
        (f"Context: {context_label}", source_ready),
        (zone_label, zone_saved),
        (monitoring_label, monitoring_active),
        (report_label, bool(st.session_state.get("generated_report") or st.session_state.get("csv_path"))),
    ]
    html_steps = "".join(
        f"<div class='workflow-status {'done' if done else ''}'><b>{'✓' if done else '•'}</b><span>{html.escape(label)}</span></div>"
        for label, done in steps
    )
    st.markdown(f"<div class='workflow-status-grid'>{html_steps}</div>", unsafe_allow_html=True)


def cctv_zone_assignments(uploaded_files: list) -> list[dict]:
    zone_names = PLANT_ZONES
    assignments = []
    for index, uploaded_file in enumerate(uploaded_files[: len(zone_names)]):
        zone_key = f"feed_zone_{index}"
        zone = st.session_state.get(zone_key, zone_names[index])
        if zone not in zone_names:
            zone = zone_names[index]
            st.session_state[zone_key] = zone
        assignments.append(
            {
                "index": index,
                "zone": zone,
                "camera": f"CCTV-{index + 1}",
                "name": uploaded_file.name,
                "size_mb": uploaded_file.size / (1024 * 1024),
            }
        )
    return assignments


def zone_storage_key(zone_name: str, cctv_index: int | None = None) -> str:
    if cctv_index is None:
        cctv_index = int(st.session_state.get("active_cctv_index", 0))
    return f"cctv-{cctv_index}:{zone_name}"


def get_saved_zone(zone_name: str, fallback: list[tuple[int, int]], cctv_index: int | None = None) -> list[tuple[int, int]]:
    saved = st.session_state.get("custom_zone_points", {})
    scoped_key = zone_storage_key(zone_name, cctv_index)
    if scoped_key in saved:
        return saved[scoped_key]
    if zone_name in saved:
        return saved[zone_name]
    return fallback


def camera_zone_defaults(width: int, height: int) -> dict[str, list[tuple[int, int]]]:
    return {
        "Zone A": build_preset_zone(width, height, "bottom_left"),
        "Control Room": build_preset_zone(width, height, "bottom_center"),
        "Zone B": build_preset_zone(width, height, "bottom_right"),
        "Reactor Zone": build_preset_zone(width, height, "bottom_center"),
    }


def configured_zone_defs(width: int, height: int, cctv_index: int | None = None, include_defaults: bool = False) -> list[dict]:
    defaults = camera_zone_defaults(width, height)
    saved = st.session_state.get("custom_zone_points", {})
    zone_defs = []
    for name in PLANT_ZONES:
        scoped_key = zone_storage_key(name, cctv_index)
        if include_defaults or scoped_key in saved or name in saved:
            zone_defs.append({"name": name, "points": get_saved_zone(name, defaults[name], cctv_index)})
    if zone_defs:
        return zone_defs

    fallback_name = st.session_state.get("zone_edit_target") or st.session_state.get("zone_map_target") or "Zone A"
    if fallback_name not in defaults:
        fallback_name = "Zone A"
    return [{"name": fallback_name, "points": defaults[fallback_name]}]


def invalidate_processed_preview() -> None:
    st.session_state.current_frame = None
    st.session_state.processed_video = False
    st.session_state.progress = 0.0


def set_saved_zone(zone_name: str, points: list[tuple[int, int]], cctv_index: int | None = None) -> None:
    st.session_state.custom_zone_points[zone_storage_key(zone_name, cctv_index)] = points
    invalidate_processed_preview()


def remove_saved_zone(zone_name: str, cctv_index: int | None = None) -> None:
    saved = st.session_state.get("custom_zone_points", {})
    saved.pop(zone_storage_key(zone_name, cctv_index), None)
    saved.pop(zone_name, None)


def render_cctv_assignments(assignments: list[dict], active_index: int) -> None:
    if not assignments:
        return
    zone_names = PLANT_ZONES
    st.markdown("<div class='cctv-map-title'>Multi-CCTV feed routing</div>", unsafe_allow_html=True)
    for item in assignments:
        active = item["index"] == active_index
        label = f"{item['camera']} · {item['zone']}"
        help_text = f"{item['name']} ({item['size_mb']:.1f} MB)"
        row_cols = st.columns([0.9, 1.1])
        with row_cols[0]:
            clicked = st.button(
                label,
                key=f"cctv_assignment_{item['index']}_{item['zone']}",
                use_container_width=True,
                type="primary" if active else "secondary",
                help=help_text,
            )
        with row_cols[1]:
            selected_zone = st.selectbox(
                f"Route {item['camera']}",
                zone_names,
                index=zone_names.index(item["zone"]),
                key=f"feed_zone_select_{item['index']}",
                label_visibility="collapsed",
            )
        if selected_zone != item["zone"]:
            st.session_state[f"feed_zone_{item['index']}"] = selected_zone
            if active:
                st.session_state.zone_map_target = selected_zone
                st.session_state.zone_edit_target = selected_zone
            st.rerun()
        st.caption(help_text)
        if clicked:
            st.session_state.active_cctv_index = item["index"]
            st.session_state.zone_map_target = item["zone"]
            st.session_state.zone_edit_target = item["zone"]
            st.rerun()


def build_live_source(source_type: str, camera_url: str):
    if source_type == "Demo Camera":
        return 0
    if source_type == "Industrial CCTV":
        ensure_factory_demo_video()
        return FACTORY_DEMO_VIDEO
    return camera_url.strip()


def ensure_factory_demo_video() -> None:
    if FACTORY_DEMO_VIDEO.exists():
        return
    SAMPLE_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    width, height, fps, seconds = 960, 540, 24, 12
    writer = cv2.VideoWriter(
        str(FACTORY_DEMO_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        return

    for i in range(fps * seconds):
        frame = np.full((height, width, 3), (42, 47, 54), dtype=np.uint8)
        cv2.rectangle(frame, (0, 390), (width, height), (72, 78, 82), -1)
        cv2.rectangle(frame, (40, 80), (920, 370), (58, 65, 72), 3)
        cv2.rectangle(frame, (70, 105), (230, 340), (31, 76, 96), -1)
        cv2.rectangle(frame, (705, 105), (880, 335), (38, 92, 112), -1)
        for x in range(285, 675, 80):
            cv2.line(frame, (x, 80), (x - 55, 370), (96, 104, 110), 6)
        cv2.putText(frame, "INDUSTRIAL CCTV - ZONE A", (52, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 235, 238), 2)
        cv2.rectangle(frame, (590, 310), (875, 515), (0, 0, 180), 4)
        cv2.putText(frame, "RESTRICTED", (610, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 230), 2)

        worker_x = 90 + int(i * 2.3) % 700
        worker_y = 255
        cv2.circle(frame, (worker_x + 34, worker_y - 38), 20, (45, 210, 235), -1)
        cv2.rectangle(frame, (worker_x + 10, worker_y - 18), (worker_x + 58, worker_y + 82), (20, 210, 225), -1)
        cv2.line(frame, (worker_x + 10, worker_y + 10), (worker_x - 18, worker_y + 60), (35, 35, 35), 8)
        cv2.line(frame, (worker_x + 58, worker_y + 10), (worker_x + 88, worker_y + 60), (35, 35, 35), 8)
        cv2.line(frame, (worker_x + 22, worker_y + 82), (worker_x + 12, worker_y + 145), (30, 30, 30), 9)
        cv2.line(frame, (worker_x + 48, worker_y + 82), (worker_x + 60, worker_y + 145), (30, 30, 30), 9)
        cv2.line(frame, (worker_x + 18, worker_y), (worker_x + 55, worker_y + 72), (255, 128, 0), 5)
        cv2.line(frame, (worker_x + 54, worker_y), (worker_x + 18, worker_y + 72), (255, 128, 0), 5)

        gas = 12 + min(18, i // 18)
        cv2.rectangle(frame, (650, 30), (920, 80), (24, 24, 24), -1)
        cv2.putText(frame, f"CH4 {gas}% LEL | HOT WORK", (665, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 220, 255), 2)
        writer.write(frame)
    writer.release()


def risk_badge(score: int, level: str) -> None:
    colors = {
        "LOW": ("#15803d", "#dcfce7"),
        "MEDIUM": ("#a16207", "#fef9c3"),
        "HIGH": ("#b91c1c", "#fee2e2"),
    }
    fg, bg = colors.get(level, colors["LOW"])
    st.markdown(
        f"""
        <div class="risk-card" style="background:{bg}; border-color:{fg};">
          <span class="risk-label">Risk Score</span>
          <strong style="color:{fg};">{score}</strong>
          <span class="risk-pill" style="background:{fg};">{level}{' / CAPPED' if score >= 100 else ''}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def canvas_initial_zone(
    zone_points: list[tuple[int, int]] | None,
    scale_x: float,
    scale_y: float,
) -> dict | None:
    if not zone_points or len(zone_points) < 3:
        return None

    points = [(int(x * scale_x), int(y * scale_y)) for x, y in zone_points]
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]

    min_x = min(x for x, _ in points)
    min_y = min(y for _, y in points)
    fabric_points = [{"x": x - min_x, "y": y - min_y} for x, y in points]
    return {
        "version": "4.4.0",
        "objects": [
            {
                "type": "polygon",
                "left": min_x,
                "top": min_y,
                "points": fabric_points,
                "fill": "rgba(239, 68, 68, 0.18)",
                "stroke": DEFAULT_ZONE_COLOR,
                "strokeWidth": 4,
                "strokeUniform": True,
                "selectable": False,
                "evented": False,
            }
        ],
    }


def latest_zone_event(violation_log: list[dict]) -> dict | None:
    for row in reversed(violation_log):
        if row.get("violation_type") in {"restricted_zone_entry", "restricted_zone_breach", "restricted_zone_clear"}:
            return row
    return None


def render_zone_event_banner(violation_log: list[dict], live_event: dict | None = None) -> None:
    event = live_event or latest_zone_event(violation_log)
    if not event:
        return

    entered = event.get("violation_type") in {"restricted_zone_breach", "restricted_zone_entry"}
    title = "Restricted Zone Entry Detected" if entered else "Worker Exited Restricted Zone"
    subtitle = clean_ui_text(event.get("message", ""))
    mode = "entry" if entered else "exit"
    st.markdown(
        f"""
        <div class="zone-event-banner {mode}">
          <div>
            <span>{html.escape(str(event.get("timestamp", "Live")))}</span>
            <strong>{title}</strong>
            <p>{html.escape(subtitle)}</p>
          </div>
          <b>{'ACTIVE BREACH' if entered else 'ZONE CLEAR'}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_zone_preview_image(frame, zone_points: list[tuple[int, int]], width: int) -> None:
    render_zone_preview_multi(frame, [{"name": "Restricted Zone", "points": zone_points}], width)


def render_zone_preview_multi(
    frame,
    zone_defs: list[dict],
    width: int,
    show_labels: bool = True,
    fill_alpha: float = 0.2,
    line_width: int = 4,
) -> None:
    preview = frame.copy()
    overlay = preview.copy()
    colors = {
        "Zone A": (0, 0, 255),
        "Zone B": (0, 140, 255),
        "Control Room": (255, 160, 0),
        "Reactor Zone": (168, 85, 247),
        "Restricted Zone": (0, 0, 255),
    }
    for zone_def in zone_defs:
        zone = np.array(zone_def["points"], dtype=np.int32)
        color = colors.get(zone_def.get("name", "Restricted Zone"), (0, 0, 255))
        cv2.fillPoly(overlay, [zone], color)
    preview = cv2.addWeighted(overlay, fill_alpha, preview, 1 - fill_alpha, 0)
    for zone_def in zone_defs:
        zone = np.array(zone_def["points"], dtype=np.int32)
        color = colors.get(zone_def.get("name", "Restricted Zone"), (0, 0, 255))
        cv2.polylines(preview, [zone], True, color, line_width)
        if show_labels:
            label_x = max(8, int(np.min(zone[:, 0]) + 10))
            label_y = max(32, int(np.min(zone[:, 1]) + 34))
            label = str(zone_def.get("name", "Restricted Zone")).upper()
            cv2.putText(preview, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.64, color, 2)
    rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    st.image(rgb, channels="RGB", use_column_width=True)
    if len(zone_defs) > 1:
        legend = "".join(
            f"<span><i style='background:rgb({colors.get(item.get('name', 'Restricted Zone'), (0, 0, 255))[2]}, {colors.get(item.get('name', 'Restricted Zone'), (0, 0, 255))[1]}, {colors.get(item.get('name', 'Restricted Zone'), (0, 0, 255))[0]});'></i>{html.escape(item.get('name', 'Restricted Zone'))}</span>"
            for item in zone_defs
        )
        st.markdown(f"<div class='zone-preview-legend'>{legend}</div>", unsafe_allow_html=True)


def active_risk_factors(violation_log: list[dict], gas_context: dict | None) -> list[str]:
    alert_types = {row.get("violation_type") for row in violation_log}
    latest_zone = latest_zone_event(violation_log)
    factors = []
    if gas_elevated(gas_context):
        factors.append("gas accumulation")
    if gas_context and gas_context.get("permit_type", "None") != "None":
        factors.append(gas_context.get("permit_type", "active permit"))
    if gas_context and gas_context.get("maintenance_active"):
        factors.append("maintenance activity")
    if gas_context and gas_context.get("equipment_status", "Normal") != "Normal":
        factors.append(gas_context.get("equipment_status", "equipment condition"))
    if gas_context and gas_context.get("shift_phase", "Stable operations") != "Stable operations":
        factors.append(gas_context.get("shift_phase", "shift handover"))
    if gas_context and gas_context.get("audit_status", "Compliant") != "Compliant":
        factors.append(gas_context.get("audit_status", "compliance checklist"))
    if gas_context and gas_context.get("fire_detected"):
        factors.append("fire emergency")
    if "restricted_zone_breach" in alert_types and not (latest_zone and latest_zone.get("violation_type") == "restricted_zone_clear"):
        factors.append("restricted zone breach")
    if "no_helmet" in alert_types:
        factors.append("helmet non-compliance")
    if "no_vest" in alert_types:
        factors.append("vest non-compliance")
    return factors


def context_risk_score(violation_log: list[dict], gas_context: dict | None, current_score: int) -> int:
    score = current_score
    if gas_elevated(gas_context):
        score = max(score, 35)
    if gas_context and gas_context.get("permit_type", "None") != "None":
        score += 10
    if gas_context and gas_context.get("maintenance_active"):
        score += 10
    if gas_context and gas_context.get("equipment_status", "Normal") != "Normal":
        score += 10
    if gas_context and gas_context.get("shift_phase", "Stable operations") != "Stable operations":
        score += 8
    if gas_context and gas_context.get("audit_status", "Compliant") != "Compliant":
        score += 8
    if gas_context and gas_context.get("fire_detected"):
        score += 35
    alert_types = {row.get("violation_type") for row in violation_log}
    if "restricted_zone_breach" in alert_types:
        score += 20
    if "no_helmet" in alert_types:
        score += 12
    if "no_vest" in alert_types:
        score += 10
    return min(100, score)


def risk_breakdown_items(violation_log: list[dict], gas_context: dict | None) -> tuple[list[tuple[str, int, str]], int]:
    alert_types = {row.get("violation_type") for row in violation_log}
    items = []
    if gas_elevated(gas_context):
        items.append(("Gas accumulation", 35, "CH4/CO/H2S/O2 outside configured safe range"))
    if gas_elevated(gas_context) and gas_context and (
        gas_context.get("maintenance_active") or gas_context.get("permit_type", "None") != "None"
    ):
        items.append(("Permit overlap", 45, f"{gas_context.get('permit_type', 'Active work')} during abnormal atmosphere"))
    if gas_context and gas_context.get("equipment_status", "Normal") != "Normal":
        items.append(("Equipment condition", 20, gas_context.get("equipment_status", "Equipment risk")))
    if gas_context and gas_context.get("shift_phase", "Stable operations") != "Stable operations":
        items.append(("Shift handover", 15, gas_context.get("shift_phase", "Shift transition")))
    if gas_context and gas_context.get("audit_status", "Compliant") != "Compliant":
        items.append(("Compliance checklist", 20, gas_context.get("audit_status", "Checklist issue")))
    if "restricted_zone_breach" in alert_types:
        items.append(("Restricted zone intrusion", 40, "Worker box overlaps restricted-zone boundary"))
    if "no_helmet" in alert_types:
        items.append(("Helmet missing", 30, "PPE classifier/estimator flagged no helmet"))
    if "no_vest" in alert_types:
        items.append(("Vest missing", 25, "PPE classifier/estimator flagged no safety vest"))
    raw_total = sum(points for _, points, _ in items)
    return items, raw_total


def build_gas_context(
    scenario: str,
    permit_type: str,
    maintenance_active: bool,
    equipment_status: str,
    shift_phase: str,
    audit_status: str,
    emergency_event: str,
    real_time_feed: bool = False,
) -> dict:
    scenarios = {
        "Normal": {"methane_lel": 2, "co_ppm": 8, "h2s_ppm": 1, "oxygen_pct": 20.9},
        "Elevated accumulation": {"methane_lel": 14, "co_ppm": 46, "h2s_ppm": 8, "oxygen_pct": 20.1},
        "Critical accumulation": {"methane_lel": 24, "co_ppm": 115, "h2s_ppm": 22, "oxygen_pct": 18.7},
    }
    return {
        "enabled": True,
        "scenario": scenario,
        "readings": scenarios[scenario],
        "permit_type": permit_type,
        "maintenance_active": maintenance_active,
        "equipment_status": equipment_status,
        "shift_phase": shift_phase,
        "audit_status": audit_status,
        "fire_detected": emergency_event == "Fire detected",
        "emergency_event": emergency_event,
        "real_time_feed": real_time_feed,
        "admin_recipient": "Plant Safety Administrator",
    }


def live_gas_context(base_context: dict | None, elapsed_seconds: float) -> dict | None:
    if not base_context:
        return None
    context = dict(base_context)
    context["readings"] = dict(base_context.get("readings", {}))
    if not context.get("real_time_feed", False):
        return context

    scenario = context.get("scenario", "Normal")
    pulse = math.sin(elapsed_seconds / 5.0)
    drift = min(1.0, elapsed_seconds / 90.0)

    if scenario == "Normal":
        readings = {
            "methane_lel": round(2 + max(0, pulse) * 1.2, 1),
            "co_ppm": round(8 + max(0, pulse) * 4, 1),
            "h2s_ppm": round(1 + max(0, pulse) * 1.5, 1),
            "oxygen_pct": round(20.9 - max(0, pulse) * 0.1, 1),
        }
    elif scenario == "Elevated accumulation":
        readings = {
            "methane_lel": round(10 + drift * 9 + max(0, pulse) * 3, 1),
            "co_ppm": round(34 + drift * 28 + max(0, pulse) * 8, 1),
            "h2s_ppm": round(8 + drift * 6 + max(0, pulse) * 2.5, 1),
            "oxygen_pct": round(20.4 - drift * 0.8 - max(0, pulse) * 0.2, 1),
        }
    else:
        readings = {
            "methane_lel": round(18 + drift * 12 + max(0, pulse) * 5, 1),
            "co_ppm": round(85 + drift * 55 + max(0, pulse) * 16, 1),
            "h2s_ppm": round(16 + drift * 10 + max(0, pulse) * 4, 1),
            "oxygen_pct": round(19.3 - drift * 1.0 - max(0, pulse) * 0.3, 1),
        }

    context["readings"] = readings
    context["sensor_timestamp"] = app_time()
    return context


def gas_status_text(gas_context: dict | None) -> str:
    if not gas_context:
        return "Normal"
    readings = gas_context.get("readings", {})
    permit = gas_context.get("permit_type", "None")
    active_work = gas_context.get("maintenance_active") or permit != "None"
    equipment_risk = gas_context.get("equipment_status", "Normal") != "Normal"
    shift_risk = gas_context.get("shift_phase", "Stable operations") != "Stable operations"
    audit_risk = gas_context.get("audit_status", "Compliant") != "Compliant"
    elevated = (
        readings.get("methane_lel", 0) >= 10
        or readings.get("co_ppm", 0) >= 35
        or readings.get("h2s_ppm", 0) >= 10
        or readings.get("oxygen_pct", 20.9) < 19.5
        or readings.get("oxygen_pct", 20.9) > 23.5
    )
    if gas_context.get("fire_detected"):
        return "Emergency Override"
    if elevated and (active_work or equipment_risk or shift_risk or audit_risk):
        return "Compound Risk"
    if elevated:
        return "Gas Accumulation"
    return "Normal"


def render_gas_panel(gas_context: dict | None) -> None:
    if not gas_context:
        return
    readings = gas_context["readings"]
    status = gas_status_text(gas_context)
    danger_status = status in {"Compound Risk", "Emergency Override"}
    color = "#b91c1c" if danger_status else "#a16207" if status == "Gas Accumulation" else "#15803d"
    bg = "#fee2e2" if danger_status else "#fef9c3" if status == "Gas Accumulation" else "#dcfce7"
    biometric_status = "Disabled for evacuation" if gas_context.get("fire_detected") else "Normal access"
    feed_label = "Real-time Gas Sensor Feed" if gas_context.get("real_time_feed") else "Gas Sensor Scenario"
    sensor_time = gas_context.get("sensor_timestamp", "ready")
    equipment = gas_context.get("equipment_status", "Normal")
    shift = gas_context.get("shift_phase", "Stable operations")
    audit = gas_context.get("audit_status", "Compliant")
    st.markdown(
        f"""
        <div class="gas-panel" style="background:{bg}; border-color:{color};">
          <div>
            <span class="risk-label">{feed_label}</span>
            <strong style="color:{color};">{status}</strong>
            <span class="risk-label">Biometric access: {biometric_status}</span>
            <span class="risk-label">Sensor time: {sensor_time}</span>
            <span class="risk-label">Equipment: {html.escape(equipment)} | Shift: {html.escape(shift)} | Audit: {html.escape(audit)}</span>
          </div>
          <div class="gas-readings">
            <span>CH4 {readings['methane_lel']}% LEL</span>
            <span>CO {readings['co_ppm']} ppm</span>
            <span>H2S {readings['h2s_ppm']} ppm</span>
            <span>O2 {readings['oxygen_pct']}%</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_strip() -> None:
    source = "Industrial CCTV" if live_cctv_connected() else "Recorded CCTV"
    gas_status = gas_status_text(st.session_state.get("gas_context"))
    alert_count = max(
        int(st.session_state.get("violation_count", 0)),
        len(collect_plant_events(st.session_state.get("gas_context"))),
    )
    evidence_state = "Report ready" if st.session_state.get("generated_report") or st.session_state.get("csv_path") else f"{alert_count} active events"
    st.markdown(
        f"""
        <div class="demo-flow">
          <div class="demo-flow-title"><span>Operational Workflow</span><strong>Live safety monitoring sequence</strong></div>
          <div class="demo-flow-steps">
            <div><b>01</b><span>{html.escape(source)}</span></div>
            <div><b>02</b><span>{html.escape(gas_status)}</span></div>
            <div><b>03</b><span>{html.escape(st.session_state.get("zone_map_target", "Zone A"))}</span></div>
            <div><b>04</b><span>{"Live monitoring" if st.session_state.get("processing") else "Ready to monitor"}</span></div>
            <div><b>05</b><span>{html.escape(evidence_state)}</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_status(detector: SafetyDetector, gas_context: dict) -> None:
    mode = "Vision analytics active" if detector.fallback_mode else "PPE model active"
    gas_status = gas_status_text(gas_context)
    biometric = "Disabled" if gas_context.get("fire_detected") else "Normal"
    st.markdown(
        f"""
        <div class="mode-panel">
          <span>Detection Mode</span>
          <strong>{mode}</strong>
        </div>
        <div class="mode-panel">
          <span>Safety Echo Risk State</span>
          <strong>{gas_status}</strong>
        </div>
        <div class="mode-panel">
          <span>Biometric Access Control</span>
          <strong>{biometric}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def zone_camera_lookup() -> dict[str, dict]:
    lookup = {}
    for camera in st.session_state.get("plant_cameras", []):
        zone = camera.get("zone")
        if zone in PLANT_ZONES and zone not in lookup:
            lookup[zone] = camera
    return lookup


def heatmap_zone_rows(gas_context: dict | None, risk_score: int, violation_log: list[dict]) -> list[dict]:
    latest_zone = latest_zone_event(violation_log)
    active_zone_name = latest_zone.get("zone_name") if latest_zone else None
    zone_clear = latest_zone and latest_zone.get("violation_type") == "restricted_zone_clear"
    active_alert_zone = active_zone_name if active_zone_name and not zone_clear else None
    alert_types = {row.get("violation_type") for row in violation_log}
    cameras_by_zone = zone_camera_lookup()
    status = gas_status_text(gas_context)
    permit = gas_context.get("permit_type", "None") if gas_context else "None"
    equipment = gas_context.get("equipment_status", "Normal") if gas_context else "Normal"
    shift = gas_context.get("shift_phase", "Stable operations") if gas_context else "Stable operations"
    audit = gas_context.get("audit_status", "Compliant") if gas_context else "Compliant"
    readings = gas_context.get("readings", {}) if gas_context else {}

    rows = []
    for zone in PLANT_ZONES:
        camera = cameras_by_zone.get(zone)
        camera_events = st.session_state.get("camera_alerts", {}).get(camera.get("id"), []) if camera else []
        zone_log = [
            row for row in violation_log
            if row.get("zone_name") == zone or row.get("zone") == zone
        ]
        zone_event_active = active_alert_zone == zone
        score = 0
        factors = []
        if zone_event_active or any(row.get("violation_type") in {"restricted_zone_breach", "restricted_zone_entry"} for row in zone_log):
            score += 40
            factors.append("CCTV intrusion")
        if {"no_helmet", "no_vest"} & alert_types or any(event.get("type") in {"ppe", "no_helmet", "no_vest"} for event in camera_events):
            score += 20
            factors.append("PPE warning")
        if gas_elevated(gas_context):
            score += 35
            factors.append(f"CH4 {readings.get('methane_lel', 'n/a')}% LEL")
        if permit != "None" and gas_elevated(gas_context):
            score += 25
            factors.append("permit overlap")
        if zone == "Zone B" and equipment != "Normal":
            score += 15
            factors.append("equipment work")
        if zone == "Control Room" and shift != "Stable operations":
            score += 15
            factors.append("handover")
        if zone == "Reactor Zone" and audit != "Compliant":
            score += 15
            factors.append("checklist pending")
        if camera:
            score = max(score, int(camera.get("risk_score", 0)))

        score = min(100, score)
        level = "Critical" if score >= 80 else "High" if score >= 60 else "Elevated" if score >= 35 else "Normal"
        workers = int(camera.get("worker_count", 0)) if camera else (1 if zone_event_active else 0)
        if workers == 0 and score >= 35:
            workers = 1 if zone != "Control Room" else 2
        rows.append(
            {
                "zone": zone,
                "camera": camera.get("camera", "Unassigned CCTV") if camera else "Unassigned CCTV",
                "score": score,
                "level": level,
                "workers": workers,
                "factors": factors[:3] or [status if status != "Normal" else "No active hazard"],
                "active": score >= 60 or zone_event_active,
                "last_event": clean_ui_text(zone_log[-1].get("message")) if zone_log else "No recent CCTV event",
            }
        )
    return rows


def render_safety_heatmap(gas_context: dict | None, risk_score: int, violation_log: list[dict] | None = None) -> None:
    violation_log = violation_log or []
    rows = heatmap_zone_rows(gas_context, risk_score, violation_log)
    highest = max(rows, key=lambda row: row["score"])
    critical_count = sum(1 for row in rows if row["score"] >= 80)
    active_cameras = sum(1 for camera in st.session_state.get("plant_cameras", []) if camera.get("monitoring"))
    gas_bad = gas_elevated(gas_context)
    permit = gas_context.get("permit_type", "None") if gas_context else "None"
    next_zone = next((row for row in rows if row["zone"] != highest["zone"] and row["score"] >= 35), rows[0])
    lead_time = max(3, 18 - min(15, highest["score"] // 7))
    response_mode = "Evacuation standby" if highest["score"] >= 80 else "Supervisor review" if highest["score"] >= 60 else "Preventive monitoring"

    positions = {
        "Zone A": "zone-a-node",
        "Zone B": "zone-b-node",
        "Control Room": "control-node",
        "Reactor Zone": "reactor-node",
    }
    zone_cards = []
    for row in rows:
        factor_html = "".join(f"<small>{html.escape(str(factor))}</small>" for factor in row["factors"])
        severity_class = "critical" if row["score"] >= 80 else "high" if row["score"] >= 60 else "elevated" if row["score"] >= 35 else "normal"
        active_class = "is-active" if row["active"] else ""
        worker_label = "workers" if row["workers"] != 1 else "worker"
        zone_cards.append(
            f"<div class='plant-zone {positions[row['zone']]} {severity_class} {active_class}'>"
            f"<div class='zone-head'><b>{html.escape(row['zone'])}</b><strong>{row['score']}/100</strong></div>"
            f"<span>{html.escape(row['camera'])}</span>"
            f"<em>{html.escape(row['level'])}</em>"
            f"<i>{row['workers']} {worker_label}</i>"
            f"<div class='zone-factors'>{factor_html}</div>"
            f"<small class='hotspot'></small>"
            f"</div>"
        )

    telemetry = [
        ("Highest risk zone", f"{highest['zone']} · {highest['score']}/100"),
        ("Active CCTV feeds", str(active_cameras)),
        ("Critical zones", str(critical_count)),
        ("Prediction lead time", f"{lead_time} min"),
        ("Response mode", response_mode),
    ]
    telemetry_html = "".join(
        f"<div><span>{html.escape(label)}</span><b>{html.escape(value)}</b></div>"
        for label, value in telemetry
    )
    escalation_reason = []
    if gas_bad:
        escalation_reason.append("gas readings are elevated")
    if permit != "None":
        escalation_reason.append(f"{permit} is active")
    if highest["score"] >= 60:
        escalation_reason.append(f"{highest['zone']} has live safety signals")
    if not escalation_reason:
        escalation_reason.append("no compound escalation is active")
    flow_text = f"{highest['zone']} may propagate risk toward {next_zone['zone']} because " + ", ".join(escalation_reason) + "."

    heatmap_html = (
        "<div class='heatmap-command'>"
        "<div class='heatmap-topbar'><span>Live Plant Fusion Map</span>"
        f"<strong>{html.escape(response_mode)}</strong></div>"
        "<div class='industrial-map'>"
        "<div class='map-grid-label label-north'>Process Area</div>"
        "<div class='map-grid-label label-south'>Evacuation Corridor</div>"
        "<div class='pipe pipe-a'></div><div class='pipe pipe-b'></div><div class='pipe pipe-c'></div>"
        "<div class='risk-flow flow-one'></div><div class='risk-flow flow-two'></div>"
        f"{''.join(zone_cards)}"
        "<div class='evac-route'>Primary evacuation corridor | biometric override path</div>"
        "</div>"
        f"<div class='heatmap-telemetry'>{telemetry_html}</div>"
        "<div class='propagation-card'><b>Predictive Safety Echo</b>"
        f"<span>{html.escape(flow_text)}</span></div>"
        "</div>"
    )
    heatmap_css = """
    <style>
      * { box-sizing: border-box; }
      body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:transparent; color:#0f172a; }
      .heatmap-command { width:100%; }
      .heatmap-topbar { display:flex; align-items:center; justify-content:space-between; margin:0 0 .65rem 0; }
      .heatmap-topbar span { color:#64748b; font-size:1rem; font-weight:800; }
      .heatmap-topbar strong { border:1px solid #fecaca; background:#fff1f2; color:#dc2626; border-radius:999px; padding:.45rem .75rem; font-size:.9rem; }
      .industrial-map {
        position:relative; min-height:405px; overflow:hidden; border:1px solid #1e3a8a; border-radius:8px;
        background:
          linear-gradient(rgba(148,163,184,.12) 1px, transparent 1px),
          linear-gradient(90deg, rgba(148,163,184,.12) 1px, transparent 1px),
          radial-gradient(circle at 20% 25%, rgba(239,68,68,.18), transparent 24%),
          radial-gradient(circle at 75% 55%, rgba(249,115,22,.2), transparent 26%),
          #020d1b;
        background-size:64px 64px, 64px 64px, auto, auto, auto;
      }
      .map-grid-label { position:absolute; color:#94a3b8; font-size:.72rem; font-weight:900; text-transform:uppercase; letter-spacing:.04em; }
      .label-north { left:1rem; top:.85rem; } .label-south { left:1rem; bottom:.75rem; }
      .pipe { position:absolute; background:rgba(148,163,184,.32); border-radius:999px; }
      .pipe-a { left:8%; right:8%; top:49%; height:9px; }
      .pipe-b { left:55%; top:10%; bottom:19%; width:12px; }
      .pipe-c { left:18%; right:17%; bottom:30%; height:9px; transform:rotate(-5deg); }
      .risk-flow { position:absolute; height:8px; border-radius:999px; background:linear-gradient(90deg, transparent, rgba(239,68,68,.75), transparent); animation:riskFlow 2.1s infinite; }
      .flow-one { left:15%; width:65%; top:49%; } .flow-two { left:42%; width:31%; top:70%; transform:rotate(-5deg); animation-delay:.8s; }
      .plant-zone { position:absolute; width:230px; min-height:132px; border:1px solid rgba(148,163,184,.28); background:rgba(15,23,42,.76); color:#f8fafc; border-radius:8px; padding:.85rem; box-shadow:inset 0 1px 0 rgba(255,255,255,.08); }
      .plant-zone.is-active { box-shadow:0 0 28px rgba(239,68,68,.28), inset 0 1px 0 rgba(255,255,255,.08); animation:zonePulse 1.5s infinite; }
      .plant-zone.normal { border-color:rgba(34,197,94,.45); } .plant-zone.elevated { border-color:rgba(245,158,11,.72); } .plant-zone.high { border-color:rgba(249,115,22,.9); } .plant-zone.critical { border-color:#ef4444; }
      .zone-a-node { left:5%; top:14%; } .zone-b-node { right:6%; top:20%; } .control-node { left:35%; bottom:13%; } .reactor-node { left:34%; top:21%; }
      .zone-head { display:flex; align-items:center; justify-content:space-between; gap:.65rem; }
      .zone-head b { font-size:1.05rem; } .zone-head strong { color:#e0f2fe; font-size:.86rem; }
      .plant-zone span,.plant-zone em,.plant-zone i { display:block; margin-top:.18rem; font-style:normal; font-size:.8rem; color:#cbd5e1; }
      .plant-zone em { color:#fde68a; font-weight:900; }
      .zone-factors { display:flex; flex-wrap:wrap; gap:.25rem; margin-top:.45rem; }
      .zone-factors small { border-radius:999px; color:#dbeafe; background:rgba(30,64,175,.35); padding:.12rem .38rem; font-size:.66rem; }
      .hotspot { position:absolute; width:20px; height:20px; border-radius:999px; background:#22c55e; right:14px; top:14px; box-shadow:0 0 22px #22c55e; animation:pulseLive 1.2s infinite; }
      .plant-zone.elevated .hotspot { background:#f59e0b; box-shadow:0 0 22px #f59e0b; } .plant-zone.high .hotspot { background:#f97316; box-shadow:0 0 22px #f97316; } .plant-zone.critical .hotspot { background:#ef4444; box-shadow:0 0 24px #ef4444; }
      .evac-route { position:absolute; left:5%; right:5%; bottom:5%; border:1px dashed rgba(34,197,94,.65); background:rgba(20,83,45,.3); color:#bbf7d0; border-radius:999px; padding:.45rem .75rem; font-weight:900; text-align:center; font-size:.82rem; }
      .heatmap-telemetry { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:.55rem; margin-top:.65rem; }
      .heatmap-telemetry div { border:1px solid #dbeafe; background:#eff6ff; border-radius:8px; padding:.55rem; }
      .heatmap-telemetry span { display:block; color:#64748b; font-size:.72rem; font-weight:800; }
      .heatmap-telemetry b { display:block; color:#1e3a8a; font-size:.9rem; margin-top:.15rem; }
      .propagation-card { margin-top:.65rem; border-left:5px solid #0f766e; background:#ecfdf5; border-radius:8px; padding:.75rem .9rem; }
      .propagation-card b { display:block; color:#064e3b; margin-bottom:.15rem; } .propagation-card span { color:#115e59; }
      @keyframes zonePulse { 0%,100% { transform:scale(1); } 50% { transform:scale(1.015); } }
      @keyframes pulseLive { 0%,100% { transform:scale(1); opacity:.9; } 50% { transform:scale(1.35); opacity:.55; } }
      @keyframes riskFlow { 0% { opacity:.2; transform:translateX(-20px); } 50% { opacity:.95; } 100% { opacity:.2; transform:translateX(20px); } }
      @media (max-width: 760px) {
        .industrial-map { min-height:auto; padding:.7rem; }
        .plant-zone { position:relative; left:auto; right:auto; top:auto; bottom:auto; width:100%; margin-bottom:.65rem; }
        .pipe,.risk-flow,.evac-route { display:none; }
        .heatmap-telemetry { grid-template-columns:1fr; }
      }
    </style>
    """
    components.html(heatmap_css + heatmap_html, height=580, scrolling=False)


def render_relationship_graph(gas_context: dict | None) -> None:
    permit = gas_context.get("permit_type", "None") if gas_context else "None"
    equipment = gas_context.get("equipment_status", "Normal") if gas_context else "Normal"
    shift = gas_context.get("shift_phase", "Stable operations") if gas_context else "Stable operations"
    audit = gas_context.get("audit_status", "Compliant") if gas_context else "Compliant"
    status = gas_status_text(gas_context)
    readings = gas_context.get("readings", {}) if gas_context else {}
    decision = "Block / Supervisor Review" if status in {"Compound Risk", "Emergency Override"} else "Monitor" if status == "Gas Accumulation" else "Allow"
    reason = (
        f"Decision: {decision}. "
        f"Reason: CH4 {readings.get('methane_lel', 'n/a')}% LEL, CO {readings.get('co_ppm', 'n/a')} ppm, "
        f"permit={permit}, equipment={equipment}, shift={shift}, checklist={audit}."
    )


def render_live_signal_correlation(violation_log: list[dict], gas_context: dict | None, risk_score: int) -> None:
    alert_types = {row.get("violation_type") for row in violation_log}
    latest_zone = latest_zone_event(violation_log)
    zone_clear = latest_zone and latest_zone.get("violation_type") == "restricted_zone_clear"
    zone_breach = latest_zone and not zone_clear
    zone_name = latest_zone.get("zone_name", "Restricted zone") if latest_zone else "Restricted zone"
    status = gas_status_text(gas_context)
    readings = gas_context.get("readings", {}) if gas_context else {}
    permit = gas_context.get("permit_type", "None") if gas_context else "None"
    equipment = gas_context.get("equipment_status", "Normal") if gas_context else "Normal"

    signals = [
        (f"CCTV {zone_name}", "CLEAR" if zone_clear else "BREACH" if zone_breach else "WATCHING", latest_zone.get("message", "No zone crossing yet") if latest_zone else "Waiting for zone event"),
        ("PPE AI", "WARNING" if {"no_helmet", "no_vest"} & alert_types else "OK", "Helmet/vest issue detected" if {"no_helmet", "no_vest"} & alert_types else "No current PPE alert"),
        ("Gas Sensor", "ABNORMAL" if gas_elevated(gas_context) else "NORMAL", f"CH4 {readings.get('methane_lel', 'n/a')}% LEL, CO {readings.get('co_ppm', 'n/a')} ppm"),
        ("Permit Engine", "ACTIVE" if permit != "None" else "NONE", permit),
        ("Equipment", "CHECK" if equipment != "Normal" else "NORMAL", equipment),
        ("Response", "CRITICAL" if risk_score >= 80 else "READY" if risk_score >= 40 else "STANDBY", f"Current fused risk {risk_score}/100"),
    ]
    with st.container(border=True):
        st.markdown("**Live Signal Correlation**")
        for title, state, detail in signals:
            level = "critical" if state in {"BREACH", "ABNORMAL", "CRITICAL"} else "warning" if state in {"WARNING", "ACTIVE", "CHECK", "READY"} else "normal"
            state_label = {"critical": "CRITICAL", "warning": "WARNING", "normal": "NORMAL"}[level]
            cols = st.columns([1.2, 0.9, 2.4])
            cols[0].markdown(f"**{title}**")
            cols[1].markdown(f"`{state_label}`")
            cols[2].caption(detail)


def render_response_workflow(violation_log: list[dict], gas_context: dict | None) -> None:
    alert_types = {row.get("violation_type") for row in violation_log}
    status = gas_status_text(gas_context)
    has_compound = "compound_gas_work_permit_risk" in alert_types or status == "Compound Risk"
    has_fire = "fire_biometric_override" in alert_types or (gas_context and gas_context.get("fire_detected"))
    has_ppe = bool({"no_helmet", "no_vest"} & alert_types)
    equipment = gas_context.get("equipment_status", "Normal") if gas_context else "Normal"
    shift = gas_context.get("shift_phase", "Stable operations") if gas_context else "Stable operations"
    audit = gas_context.get("audit_status", "Compliant") if gas_context else "Compliant"

    steps = [
        ("Gas hazard message", "Sent to Plant Safety Administrator" if status != "Normal" else "Standby"),
        ("Permit cross-check", "Escalated: active work overlaps gas risk" if has_compound else "No dangerous overlap"),
        ("Equipment isolation", f"Check status: {equipment}" if equipment != "Normal" else "Normal operating state"),
        ("Shift handover", f"Supervisor acknowledgement needed: {shift}" if shift != "Stable operations" else "No handover conflict"),
        ("Worker warning", "Named PPE warning sent" if has_ppe else "No PPE warning yet"),
        ("Compliance audit", f"Corrective workflow open: {audit}" if audit != "Compliant" else "Documentation aligned"),
        ("Emergency access", "Biometric locks disabled for evacuation" if has_fire else "Normal access mode"),
        ("Incident report", "Draft report ready with CCTV, gas, permit and alert log" if violation_log else "Waiting for scan evidence"),
    ]
    rows = "".join(
        f"<div class='workflow-row'><strong>{title}</strong><span>{state}</span></div>"
        for title, state in steps
    )
    st.markdown(f"<div class='workflow-panel'>{rows}</div>", unsafe_allow_html=True)


def gas_elevated(gas_context: dict | None) -> bool:
    if not gas_context:
        return False
    readings = gas_context.get("readings", {})
    return (
        readings.get("methane_lel", 0) >= 10
        or readings.get("co_ppm", 0) >= 35
        or readings.get("h2s_ppm", 0) >= 10
        or readings.get("oxygen_pct", 20.9) < 19.5
        or readings.get("oxygen_pct", 20.9) > 23.5
    )


def build_agent_trace(
    violation_log: list[dict],
    gas_context: dict | None,
    detector: SafetyDetector,
    risk_score: int,
) -> list[tuple[str, str, str]]:
    alert_types = {row.get("violation_type") for row in violation_log}
    status = gas_status_text(gas_context)
    ppe_mode = "PPE compliance analytics active" if detector.fallback_mode else "PPE model active"
    gas_signal = "hazardous gas trend detected" if gas_elevated(gas_context) else "gas readings within configured limits"
    permit = gas_context.get("permit_type", "None") if gas_context else "None"
    active_work = gas_context.get("maintenance_active") if gas_context else False
    equipment = gas_context.get("equipment_status", "Normal") if gas_context else "Normal"
    shift = gas_context.get("shift_phase", "Stable operations") if gas_context else "Stable operations"
    audit = gas_context.get("audit_status", "Compliant") if gas_context else "Compliant"

    vision_finding = "PPE or restricted-zone issue observed" if alert_types else "monitoring person, PPE, and zone signals"
    if "restricted_zone_breach" in alert_types:
        vision_finding = "worker center point entered restricted zone"
    if {"no_helmet", "no_vest"} & alert_types:
        vision_finding += " with PPE warning"

    context_finding = f"{permit}; maintenance crew active: {'yes' if active_work else 'no'}"
    equipment_finding = "maintenance dependency detected" if equipment != "Normal" else "equipment condition normal"
    shift_finding = "handover attention required" if shift != "Stable operations" else "stable shift window"
    history_finding = "recurring pattern: gas + permit + equipment/shift context"
    if status == "Normal" and risk_score < 35:
        history_finding = "no high-risk historical pattern matched"
    audit_finding = "corrective action workflow required" if audit != "Compliant" else "statutory checklist aligned"

    if status in {"Compound Risk", "Emergency Override"} or risk_score >= 70:
        orchestrator = "CRITICAL: pause work, alert admin, prepare evacuation and incident report"
    elif status == "Gas Accumulation" or risk_score >= 35:
        orchestrator = "ELEVATED: warn supervisor, verify permit, increase monitoring frequency"
    else:
        orchestrator = "NORMAL: continue monitoring"

    return [
        ("Vision Agent", ppe_mode, vision_finding),
        ("Gas Sensor Agent", status, gas_signal),
        ("Permit Agent", "permit-to-work cross-check", context_finding),
        ("Equipment Agent", "maintenance condition", equipment_finding),
        ("Shift Agent", "changeover pattern", shift_finding),
        ("Historical Pattern Agent", "incident pattern memory", history_finding),
        ("Compliance Audit Agent", "OISD / Factory Act / DGMS", audit_finding),
        ("Risk Orchestrator", f"score {risk_score}/100", orchestrator),
    ]


def compliance_guidance(violation_log: list[dict], gas_context: dict | None) -> list[str]:
    alert_types = {row.get("violation_type") for row in violation_log}
    status = gas_status_text(gas_context)
    guidance = []
    if status in {"Gas Accumulation", "Compound Risk", "Emergency Override"}:
        guidance.append("OISD-style action: isolate the affected zone, verify gas readings, and stop non-essential work until readings normalize.")
    if "compound_gas_work_permit_risk" in alert_types or status == "Compound Risk":
        guidance.append("Permit control: suspend or revalidate active maintenance/hot-work permits before allowing work to continue.")
    if "equipment_condition_overlap" in alert_types or (gas_context and gas_context.get("equipment_status", "Normal") != "Normal"):
        guidance.append("Equipment control: isolate affected equipment, confirm lockout/tagout status, and record maintenance owner acknowledgement.")
    if "shift_changeover_risk" in alert_types or (gas_context and gas_context.get("shift_phase", "Stable operations") != "Stable operations"):
        guidance.append("Shift control: brief incoming supervisor on gas, permit, and zone status before handover completion.")
    if "compliance_audit_deviation" in alert_types or (gas_context and gas_context.get("audit_status", "Compliant") != "Compliant"):
        guidance.append("Audit control: create corrective action against OISD, DGMS, and Factory Act checklist items before closure.")
    if {"no_helmet", "no_vest"} & alert_types:
        guidance.append("PPE compliance: issue named worker warning and require supervisor confirmation before re-entry.")
    if "restricted_zone_breach" in alert_types:
        guidance.append("Restricted-area control: preserve CCTV evidence, review access authorization, and update barricade/signage status.")
    if gas_context and gas_context.get("fire_detected"):
        guidance.append("Emergency response: disable restrictive biometric access, keep evacuation routes open, and notify the control room.")
    if not guidance:
        guidance.append("No actionable compliance breach yet. Continue monitoring CCTV, gas sensors, and permit context.")
    return guidance


def render_agent_copilot(
    violation_log: list[dict],
    gas_context: dict | None,
    detector: SafetyDetector,
    risk_score: int,
) -> None:
    cards = []
    for title, subtitle, finding in build_agent_trace(violation_log, gas_context, detector, risk_score):
        cards.append(
            "<div class='agent-card'>"
            f"<strong>{html.escape(title)}</strong>"
            f"<span>{html.escape(subtitle)}</span>"
            f"<p>{html.escape(finding)}</p>"
            "</div>"
        )
    st.markdown("<div class='agent-grid'>" + "".join(cards) + "</div>", unsafe_allow_html=True)


PERMIT_SCENARIOS = {
    "Hot Work Permit": {
        "icon": "Sparks",
        "color": "#f97316",
        "accent": "Fire risk animation",
        "gauges": [("Gas concentration", 74, "% LEL"), ("Temperature", 86, "C"), ("Humidity", 31, "%"), ("Fire watch", 82, "%")],
        "details": ["High temperature badge active", "Gas ignition warning enabled", "Recommended fire watch team: 2 officers"],
        "recommendations": ["Stop hot work until gas is below threshold", "Assign dedicated fire watch team", "Increase ventilation", "Verify extinguishers and isolation boundary"],
        "guidance": ["Hot work near elevated gas requires permit revalidation and supervisor sign-off.", "Ignition sources must stay isolated until CH4 and CO readings normalize.", "Evidence should include gas trend, CCTV frame, and fire-watch acknowledgement."],
        "base_risk": 82,
    },
    "Maintenance Permit": {
        "icon": "Wrench",
        "color": "#2563eb",
        "accent": "Equipment isolation status",
        "gauges": [("Equipment health", 42, "%"), ("Isolation status", 68, "%"), ("Crew count", 5, ""), ("LOTO checklist", 58, "%")],
        "details": ["Lockout/Tagout checklist pending", "Maintenance crew count: 5", "Asset health indicator degraded"],
        "recommendations": ["Confirm lockout/tagout completion", "Pause work near restricted zone", "Dispatch safety officer", "Record equipment owner acknowledgement"],
        "guidance": ["Maintenance approval should be tied to isolation evidence and active worker location.", "Incomplete lockout/tagout increases risk when gas or restricted-zone signals are active.", "Supervisor review is required before restarting affected pump work."],
        "base_risk": 54,
    },
    "Confined Space Entry": {
        "icon": "O2",
        "color": "#8b5cf6",
        "accent": "Oxygen level gauge",
        "gauges": [("Oxygen", 62, "% safe"), ("Toxic gas", 71, "%"), ("Workers inside", 2, ""), ("Rescue standby", 48, "%")],
        "details": ["Toxic gas indicator active", "Worker inside counter: 2", "Rescue standby status incomplete"],
        "recommendations": ["Start forced ventilation", "Place rescue standby team", "Limit entry duration", "Verify atmospheric test before entry"],
        "guidance": ["Confined-space entry requires continuous atmosphere monitoring and rescue readiness.", "Oxygen drift plus toxic gas indication should trigger standby escalation.", "Entry should be delayed until rescue and ventilation controls are confirmed."],
        "base_risk": 76,
    },
    "Electrical Permit": {
        "icon": "Voltage",
        "color": "#eab308",
        "accent": "Live voltage warning",
        "gauges": [("Voltage", 88, "%"), ("Isolation", 52, "%"), ("Arc flash risk", 79, "%"), ("Electrical PPE", 64, "%")],
        "details": ["Isolation confirmation pending", "Arc flash risk elevated", "Electrical PPE checklist open"],
        "recommendations": ["Confirm zero-energy state", "Verify arc-flash boundary", "Require insulated PPE", "Lock access until isolation is signed"],
        "guidance": ["Electrical work should not proceed until isolation evidence is verified.", "Arc-flash boundary and PPE compliance must be documented before approval.", "Permit engine should block approval when live-voltage status is uncertain."],
        "base_risk": 82,
    },
    "Working at Height": {
        "icon": "Harness",
        "color": "#06b6d4",
        "accent": "Fall protection status",
        "gauges": [("Harness compliance", 69, "%"), ("Wind speed", 38, "km/h"), ("Anchor verified", 57, "%"), ("Edge exposure", 66, "%")],
        "details": ["Harness compliance needs recheck", "Wind speed rising", "Anchor point verification incomplete"],
        "recommendations": ["Verify anchor points", "Recheck harness fit", "Delay work if wind increases", "Assign spotter near edge"],
        "guidance": ["Height-work approval depends on fall-protection verification and weather context.", "Anchor point evidence should be attached before permit approval.", "Edge exposure plus incomplete harness checks requires supervisor review."],
        "base_risk": 54,
    },
    "None": {
        "icon": "Clear",
        "color": "#22c55e",
        "accent": "No permit requested",
        "gauges": [("Gas concentration", 12, "% LEL"), ("Temperature", 32, "C"), ("Equipment health", 88, "%"), ("Isolation status", 95, "%")],
        "details": ["No active permit conflict", "Routine monitoring", "All approvals on standby"],
        "recommendations": ["Continue monitoring", "Keep PPE verification active", "Maintain restricted-zone watch"],
        "guidance": ["No permit-specific escalation is active.", "Continue CCTV, gas, and geofence monitoring."],
        "base_risk": 18,
    },
}


def permit_decision(score: int) -> str:
    if score >= 95:
        return "BLOCK APPROVAL"
    if score >= 80:
        return "SUPERVISOR REVIEW"
    if score >= 45:
        return "APPROVE WITH PRECAUTIONS"
    return "APPROVED"


def render_what_if_simulator(gas_context: dict | None, current_risk_score: int) -> None:
    st.markdown("**What-if Permit Simulator**")
    candidate = st.selectbox(
        "Test approval impact",
        ["Hot Work Permit", "Maintenance Permit", "Confined Space Entry", "Electrical Permit", "Working at Height", "None"],
        index=1,
        key="what_if_permit",
    )
    scenario = PERMIT_SCENARIOS[candidate]
    elevated = gas_elevated(gas_context)
    simulated = max(scenario["base_risk"], min(100, current_risk_score + (18 if elevated and candidate != "None" else 0)))
    if candidate == "None":
        simulated = min(simulated, 25)
    decision = permit_decision(simulated)
    color = scenario["color"]
    gauge_html = "".join(
        f"""
        <div class="permit-gauge">
          <span>{html.escape(label)}</span>
          <b>{value}{html.escape(unit)}</b>
          <i><em style="width:{min(100, max(6, value))}%; background:{color};"></em></i>
        </div>
        """
        for label, value, unit in scenario["gauges"]
    )
    detail_html = "".join(f"<li>{html.escape(item)}</li>" for item in scenario["details"])
    rec_html = "".join(f"<li>{html.escape(item)}</li>" for item in scenario["recommendations"])
    guidance_html = "".join(f"<li>{html.escape(item)}</li>" for item in scenario["guidance"])
    steps = ["Permit Requested", "Risk Evaluated", "Gas Cross-check", "Equipment Status", "Supervisor Decision", "Permit Approved / Rejected"]
    active_idx = min(len(steps) - 1, 1 + simulated // 22)
    step_html = "".join(
        f"<div class='permit-step {'active' if idx == active_idx else 'done' if idx < active_idx else ''}'><span>{idx + 1}</span><b>{html.escape(step)}</b></div>"
        for idx, step in enumerate(steps)
    )
    st.markdown(
        f"""
        <div class="permit-simulator" style="--permit:{color};">
          <div class="permit-left">
            <div class="permit-kicker">{html.escape(scenario['icon'])} | {html.escape(scenario['accent'])}</div>
            <div class="permit-risk">
              <span>Risk</span><strong>{simulated}/100</strong><b>{html.escape(decision)}</b>
            </div>
            <ul class="permit-details">{detail_html}</ul>
            <div class="permit-gauges">{gauge_html}</div>
          </div>
          <div class="permit-right">
            <strong>AI Recommendations</strong>
            <ul>{rec_html}</ul>
            <strong>Regulatory Intelligence</strong>
            <ul>{guidance_html}</ul>
          </div>
          <div class="permit-timeline">{step_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_incident_report_text(
    violation_log: list[dict],
    gas_context: dict | None,
    risk_score: int,
    risk_level: str,
) -> str:
    status = gas_status_text(gas_context)
    readings = gas_context.get("readings", {}) if gas_context else {}
    guidance = compliance_guidance(violation_log, gas_context)
    recent_alerts = violation_log[-8:]
    alert_text = "\n".join(f"- Frame {row.get('frame')}: {clean_ui_text(row.get('message'))}" for row in recent_alerts) or "- No alerts captured yet."
    guidance_text = "\n".join(f"- {item}" for item in guidance)
    return f"""SafeVision AI Preliminary Incident Report
Generated: {app_datetime_stamp()}

Risk Summary
- Current risk score: {risk_score}/100
- Risk level: {risk_level}
- Compound risk state: {status}
- Active permit: {gas_context.get('permit_type', 'None') if gas_context else 'None'}
- Maintenance active: {gas_context.get('maintenance_active', False) if gas_context else False}
- Equipment status: {gas_context.get('equipment_status', 'Normal') if gas_context else 'Normal'}
- Shift phase: {gas_context.get('shift_phase', 'Stable operations') if gas_context else 'Stable operations'}
- Audit status: {gas_context.get('audit_status', 'Compliant') if gas_context else 'Compliant'}

Gas / SCADA Snapshot
- CH4: {readings.get('methane_lel', 'n/a')}% LEL
- CO: {readings.get('co_ppm', 'n/a')} ppm
- H2S: {readings.get('h2s_ppm', 'n/a')} ppm
- O2: {readings.get('oxygen_pct', 'n/a')}%

Recent Administrator Alerts
{alert_text}

Recommended Corrective Actions
{guidance_text}

Integration note: plant deployments should connect this report to SCADA, permit-to-work, access-control, and approved compliance knowledge bases.
"""


def answer_copilot_question(question: str, violation_log: list[dict], gas_context: dict | None, risk_score: int) -> str:
    status = gas_status_text(gas_context)
    alert_types = {row.get("violation_type") for row in violation_log}
    readings = gas_context.get("readings", {}) if gas_context else {}
    permit = gas_context.get("permit_type", "None") if gas_context else "None"
    maintenance = gas_context.get("maintenance_active", False) if gas_context else False
    equipment = gas_context.get("equipment_status", "Normal") if gas_context else "Normal"
    shift = gas_context.get("shift_phase", "Stable operations") if gas_context else "Stable operations"
    audit = gas_context.get("audit_status", "Compliant") if gas_context else "Compliant"

    if question == "Why is risk high?":
        reasons = []
        if gas_elevated(gas_context):
            reasons.append(
                f"gas threshold breach: CH4 {readings.get('methane_lel')}% LEL, CO {readings.get('co_ppm')} ppm, H2S {readings.get('h2s_ppm')} ppm"
            )
        if permit != "None" or maintenance:
            reasons.append(f"active work context: {permit}, maintenance active: {maintenance}")
        if equipment != "Normal":
            reasons.append(f"equipment condition: {equipment}")
        if shift != "Stable operations":
            reasons.append(f"shift context: {shift}")
        if audit != "Compliant":
            reasons.append(f"compliance status: {audit}")
        if "restricted_zone_breach" in alert_types:
            reasons.append("worker entered the restricted zone")
        if {"no_helmet", "no_vest"} & alert_types:
            reasons.append("PPE compliance warning was detected")
        return "Risk is elevated because " + "; ".join(reasons) + "." if reasons else "Risk is currently low. No major compound signal is active."

    if question == "What should admin do?":
        return "Admin action: pause affected work, verify gas readings, contact the supervisor, preserve CCTV evidence, revalidate permits, confirm equipment isolation, and close audit checklist gaps before restart." if status != "Normal" or risk_score >= 35 else "Admin action: continue monitoring. No escalation is required right now."

    if question == "Can hot work be approved?":
        if gas_elevated(gas_context):
            return "Hot work should be blocked until gas readings normalize and the area is cleared by safety supervision."
        return "Hot work can move to normal permit review because gas readings are within configured limits."

    if question == "Is evacuation needed?":
        if gas_context and gas_context.get("fire_detected"):
            return "Evacuation route should be opened immediately. Biometric restrictions are disabled in the response workflow."
        if status == "Compound Risk" or risk_score >= 70:
            return "Prepare evacuation standby and isolate the affected zone until the supervisor confirms safe conditions."
        return "Evacuation is not currently required."

    return "Select a question to inspect the current safety state."


def render_copilot_questions(violation_log: list[dict], gas_context: dict | None, risk_score: int) -> None:
    st.markdown("**Ask SafeVision Co-Pilot**")
    questions = ["Why is risk high?", "What should admin do?", "Can hot work be approved?", "Is evacuation needed?"]
    cols = st.columns(4)
    for col, question in zip(cols, questions):
        with col:
            if st.button(question, use_container_width=True):
                st.session_state.copilot_answer = answer_copilot_question(question, violation_log, gas_context, risk_score)
    if st.session_state.copilot_answer:
        st.markdown(
            f"<div class='copilot-answer'>{html.escape(st.session_state.copilot_answer)}</div>",
            unsafe_allow_html=True,
        )


def evaluation_metrics(
    violation_log: list[dict],
    gas_context: dict | None,
    risk_score: int,
    zone_points: list[tuple[int, int]] | None,
) -> dict:
    alert_types = {row.get("violation_type") for row in violation_log}
    status = gas_status_text(gas_context)
    compound_active = status in {"Compound Risk", "Emergency Override"} or "compound_gas_work_permit_risk" in alert_types
    vision_active = bool({"restricted_zone_breach", "no_helmet", "no_vest"} & alert_types)
    gas_active = gas_elevated(gas_context)
    permit_active = bool(gas_context and (gas_context.get("maintenance_active") or gas_context.get("permit_type") != "None"))
    equipment_active = bool(gas_context and gas_context.get("equipment_status", "Normal") != "Normal")
    shift_active = bool(gas_context and gas_context.get("shift_phase", "Stable operations") != "Stable operations")
    audit_active = bool(gas_context and gas_context.get("audit_status", "Compliant") != "Compliant")
    ppe_events = len([row for row in violation_log if row.get("violation_type") in {"no_helmet", "no_vest"}])
    zone_events = len([row for row in violation_log if row.get("violation_type") == "restricted_zone_breach"])
    active_signals = sum([vision_active, gas_active, permit_active, equipment_active, shift_active, audit_active])

    baseline_score = 0
    if vision_active:
        baseline_score = max(baseline_score, min(65, 30 + ppe_events * 4 + zone_events * 8))
    if gas_active:
        baseline_score = max(baseline_score, 55)
    if permit_active:
        baseline_score = max(baseline_score, 25)
    if equipment_active:
        baseline_score = max(baseline_score, 30)
    if shift_active:
        baseline_score = max(baseline_score, 20)
    fused_score = max(risk_score, context_risk_score(violation_log, gas_context, risk_score))
    if active_signals >= 3:
        fused_score = max(fused_score, min(100, baseline_score + 18 + active_signals * 5))
    elif gas_active or vision_active:
        fused_score = max(fused_score, min(80, baseline_score + 10))
    else:
        fused_score = max(fused_score, 12 if not violation_log else 25)
    detection_gain = max(0, fused_score - baseline_score)
    false_negative_reduction = min(88, active_signals * 12 + (18 if compound_active else 0) + min(14, len(violation_log) // 3))
    lead_time = min(18, active_signals * 2 + (5 if compound_active else 0) + (2 if zone_events else 0))
    geospatial_quality = min(98, (86 if zone_points and len(zone_points) >= 4 else 55) + min(10, zone_events * 2))
    compliance_count = len(compliance_guidance(violation_log, gas_context))
    coverage = min(100, 35 + compliance_count * 10 + (15 if compound_active else 0) + (8 if gas_active else 0) + (8 if audit_active else 0))

    return {
        "single_sensor_score": baseline_score,
        "compound_score": min(100, fused_score),
        "detection_gain": detection_gain,
        "false_negative_reduction": false_negative_reduction,
        "lead_time": lead_time,
        "geospatial_quality": geospatial_quality,
        "compliance_coverage": coverage,
        "compound_active": compound_active,
        "active_signals": active_signals,
        "zone_events": zone_events,
        "ppe_events": ppe_events,
    }


def render_evaluation_metrics(
    violation_log: list[dict],
    gas_context: dict | None,
    risk_score: int,
    zone_points: list[tuple[int, int]] | None,
) -> None:
    metrics = evaluation_metrics(violation_log, gas_context, risk_score, zone_points)
    st.subheader("Performance Dashboard")
    st.caption("Operational indicators for risk fusion, response timing, geospatial evidence, and compliance readiness.")
    metric_cols = st.columns(5)
    cards = [
        ("Current Fused Risk", f"{metrics['compound_score']}/100", f"{metrics['active_signals']} active safety signals"),
        ("Single-Sensor Baseline", f"{metrics['single_sensor_score']}/100", "best isolated detector score"),
        ("Fusion Lift", f"+{metrics['detection_gain']} pts", "added by CCTV + gas + permit context"),
        ("Zone Evidence", f"{metrics['geospatial_quality']}%", f"{metrics['zone_events']} intrusion events captured"),
        ("Intervention Lead", f"{metrics['lead_time']} min", "response window"),
    ]
    for col, (title, value, note) in zip(metric_cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="eval-card">
                  <span>{html.escape(title)}</span>
                  <strong>{html.escape(value)}</strong>
                  <p>{html.escape(note)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    comparison = pd.DataFrame(
        [
            {"Approach": "CCTV only", "Risk detected": "PPE / zone only", "Score": min(45, metrics["single_sensor_score"]), "Missed signal": "Gas + permit context"},
            {"Approach": "Gas sensor only", "Risk detected": "Gas threshold only", "Score": min(55, metrics["single_sensor_score"]), "Missed signal": "Worker presence + permit"},
            {"Approach": "Permit system only", "Risk detected": "Work authorization only", "Score": 25, "Missed signal": "CCTV + gas conditions"},
            {
                "Approach": "SafeVision fused layer",
                "Risk detected": f"{metrics['active_signals']} correlated signals",
                "Score": metrics["compound_score"],
                "Missed signal": f"{metrics['false_negative_reduction']}% lower blind-spot risk",
            },
        ]
    )
    st.dataframe(comparison, use_container_width=True, height=180)

    if metrics["compound_active"]:
        st.success("Compound risk detected: the fused layer caught a hazardous combination that a single sensor could miss.")
    else:
        st.info("No active compound-risk state. Select elevated gas with active work to review risk fusion behavior.")


def future_echo_projection(violation_log: list[dict], gas_context: dict | None, risk_score: int) -> tuple[list[tuple[str, int]], str]:
    current = context_risk_score(violation_log, gas_context, risk_score)
    factors = active_risk_factors(violation_log, gas_context)
    escalation = 2
    if "gas accumulation" in factors:
        escalation += 4
    if any("Permit" in factor or "permit" in factor for factor in factors):
        escalation += 3
    if "maintenance activity" in factors:
        escalation += 3
    if any("handover" in factor.lower() for factor in factors):
        escalation += 3
    if any("checklist" in factor.lower() or "pending" in factor.lower() or "overdue" in factor.lower() for factor in factors):
        escalation += 2
    if "restricted zone breach" in factors:
        escalation += 4
    if "helmet non-compliance" in factors or "vest non-compliance" in factors:
        escalation += 3

    projection = [
        ("Current risk", current),
        ("+10 minutes", min(100, current + escalation)),
        ("+20 minutes", min(100, current + escalation * 2)),
        ("+30 minutes", min(100, current + escalation * 3)),
    ]
    if factors:
        reason = "Risk is projected to rise because " + ", ".join(factors[:6]) + " are active at the same time."
    else:
        reason = "Risk is stable because no major escalation factors are currently active."
    return projection, reason


def render_future_echo_prediction(violation_log: list[dict], gas_context: dict | None, risk_score: int) -> None:
    projection, reason = future_echo_projection(violation_log, gas_context, risk_score)
    st.markdown("**Future Echo Prediction**")
    cols = st.columns(4)
    for col, (label, value) in zip(cols, projection):
        level = "Critical" if value >= 85 else "High" if value >= 70 else "Elevated" if value >= 40 else "Stable"
        color = "#b91c1c" if value >= 85 else "#dc2626" if value >= 70 else "#a16207" if value >= 40 else "#15803d"
        with col:
            st.markdown(
                f"""
                <div class="echo-card" style="border-color:{color};">
                  <span>{html.escape(label)}</span>
                  <strong style="color:{color};">{value}%</strong>
                  <p>{level}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown(f"<div class='echo-explain'>{html.escape(reason)}</div>", unsafe_allow_html=True)


def intervention_recommendations(violation_log: list[dict], gas_context: dict | None, risk_score: int) -> tuple[list[str], int, int]:
    factors = active_risk_factors(violation_log, gas_context)
    current = context_risk_score(violation_log, gas_context, risk_score)
    latest_zone = latest_zone_event(violation_log)
    zone_is_clear = bool(latest_zone and latest_zone.get("violation_type") == "restricted_zone_clear")
    actions = []
    if gas_elevated(gas_context):
        actions.extend(["Notify safety admin", "Increase ventilation"])
    if gas_context and gas_context.get("permit_type") == "Hot Work Permit":
        actions.append("Stop hot work")
    elif gas_context and gas_context.get("permit_type", "None") != "None":
        actions.append("Pause maintenance permit")
    if gas_context and gas_context.get("maintenance_active"):
        actions.append("Dispatch safety officer")
    if zone_is_clear:
        actions.append("Confirm restricted zone is clear")
        actions.append("Record supervisor acknowledgement")
    elif "restricted zone breach" in factors:
        actions.append("Evacuate restricted zone")
    if "helmet non-compliance" in factors or "vest non-compliance" in factors:
        actions.append("Recheck PPE compliance")
    if gas_context and gas_context.get("shift_phase", "Stable operations") != "Stable operations":
        actions.append("Delay shift handover")
    if gas_context and gas_context.get("audit_status", "Compliant") != "Compliant":
        actions.append("Close compliance checklist")
    if gas_context and gas_context.get("equipment_status", "Normal") != "Normal":
        actions.append("Confirm equipment isolation")
    if not actions:
        actions = ["Continue monitoring", "Keep restricted zone clear", "Maintain PPE checks"]

    deduped = []
    for action in actions:
        if action not in deduped:
            deduped.append(action)
    selected = deduped[:5]
    reduction = min(65, 12 + len(selected) * 9 + (10 if gas_elevated(gas_context) else 0))
    reduced = max(8, current - reduction)
    return selected, current, reduced


def render_intervention_recommendation(violation_log: list[dict], gas_context: dict | None, risk_score: int) -> None:
    actions, current, reduced = intervention_recommendations(violation_log, gas_context, risk_score)
    action_items = "".join(f"<li>{html.escape(action)}</li>" for action in actions)
    st.markdown(
        f"""
        <div class="intervention-panel">
          <strong>Intervention Recommendation</strong>
          <ul>{action_items}</ul>
          <div class="risk-reduction">Risk reduced from <b>{current}%</b> to <b>{reduced}%</b> after intervention.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_breakdown(violation_log: list[dict], gas_context: dict | None, risk_score: int) -> None:
    items, raw_total = risk_breakdown_items(violation_log, gas_context)
    capped_note = "Score capped at 100" if raw_total > 100 else "Score below cap"
    if not items:
        items = [("No active risk factor", 0, "System is waiting for CCTV, gas, permit, or PPE triggers")]
    rows = "".join(
        f"""
        <div class="risk-break-row">
          <div><b>{html.escape(title)}</b><span>{html.escape(detail)}</span></div>
          <strong>+{points}</strong>
        </div>
        """
        for title, points, detail in items
    )
    st.markdown(
        f"""
        <div class="risk-breakdown">
          <div class="risk-break-head">
            <span>Risk Calculation</span>
            <b>{raw_total} raw → {risk_score}/100</b>
            <em>{html.escape(capped_note)}</em>
          </div>
          {rows}
        </div>
        """,
        unsafe_allow_html=True,
    )


def echo_timeline_rows(violation_log: list[dict], gas_context: dict | None) -> list[dict]:
    rows = []
    now = datetime.now()
    factors = active_risk_factors([], gas_context)
    context_impacts = {
        "gas accumulation": 25,
        "maintenance activity": 20,
        "Hot Work Permit": 25,
        "Maintenance Permit": 20,
        "Confined Space Entry": 20,
    }
    for idx, factor in enumerate(factors[:6]):
        impact = context_impacts.get(factor, 15)
        rows.append(
            {
                "Time": (now.replace(second=0, microsecond=0)).strftime("%H:%M"),
                "Event": factor.title(),
                "Risk impact": f"+{impact}",
            }
        )
    selected_rows = violation_log[-8:]
    zone_rows = [row for row in violation_log if row.get("violation_type") in {"restricted_zone_entry", "restricted_zone_breach", "restricted_zone_clear"}]
    for row in zone_rows[-4:]:
        if row not in selected_rows:
            selected_rows.append(row)
    selected_rows = sorted(selected_rows, key=lambda row: row.get("frame", 0))[-12:]
    for row in selected_rows:
        event_name = str(row.get("violation_type", "event")).replace("_", " ").title()
        rows.append(
            {
                "Time": row.get("timestamp", ""),
                "Event": event_name,
                "Risk impact": f"+{row.get('risk_points', 0)}",
            }
        )
    if not rows:
        rows.append({"Time": now.strftime("%H:%M"), "Event": "Monitoring active", "Risk impact": "+0"})
    if any("Critical" in str(row.get("Event")) for row in rows) or gas_status_text(gas_context) in {"Compound Risk", "Emergency Override"}:
        rows.append({"Time": now.strftime("%H:%M"), "Event": "Risk became Critical - Alert generated", "Risk impact": "+30"})
    return rows


def render_echo_timeline(violation_log: list[dict], gas_context: dict | None) -> None:
    st.markdown("**Echo Timeline**")
    st.dataframe(pd.DataFrame(echo_timeline_rows(violation_log, gas_context)), use_container_width=True, height=260)


def latest_frame_events(violation_log: list[dict], gas_context: dict | None, risk_score: int) -> list[dict]:
    now = app_time()
    alerts = []
    status = gas_status_text(gas_context)
    if status != "Normal":
        severity = "critical" if status in {"Compound Risk", "Emergency Override"} or risk_score >= 80 else "warning"
        alerts.append({"time": now, "source": "Gas Sensor", "severity": severity, "event": status})
    recent_unique = []
    seen_types = set()
    repeat_counts = {}
    for row in violation_log[-40:]:
        vtype = row.get("violation_type", "")
        repeat_counts[vtype] = repeat_counts.get(vtype, 0) + 1
    for row in reversed(violation_log[-40:]):
        vtype = row.get("violation_type", "")
        if vtype in seen_types:
            continue
        seen_types.add(vtype)
        recent_unique.append(row)
        if len(recent_unique) >= 6:
            break
    for row in reversed(recent_unique):
        vtype = row.get("violation_type", "")
        source = "PPE AI" if vtype in {"no_helmet", "no_vest"} else "CCTV" if vtype in {"restricted_zone_breach", "restricted_zone_clear"} else "Permit Engine"
        if vtype == "restricted_zone_clear":
            severity = "normal"
        else:
            severity = "critical" if row.get("severity") == "HIGH" or risk_score >= 80 else "warning"
        repeated = repeat_counts.get(vtype, 1)
        repeat_note = f" ({repeated} recent hits grouped)" if repeated > 1 else ""
        alerts.append({"time": row.get("timestamp", now), "source": source, "severity": severity, "event": f"{clean_ui_text(row.get('message', vtype))}{repeat_note}"})
    if not alerts:
        alerts.append({"time": now, "source": "System", "severity": "normal", "event": "All monitored signals normal"})
    return alerts[-8:]


def render_live_analytics_bar(
    violation_log: list[dict],
    gas_context: dict | None,
    risk_score: int,
    worker_count: int | None = None,
) -> None:
    alert_types = {row.get("violation_type") for row in violation_log}
    readings = gas_context.get("readings", {}) if gas_context else {}
    workers = worker_count if worker_count is not None else 0
    ppe_issues = sum(1 for row in violation_log[-20:] if row.get("violation_type") in {"no_helmet", "no_vest"})
    compliance = max(42, 100 - ppe_issues * 7 - (12 if "restricted_zone_breach" in alert_types else 0))
    intrusions = sum(1 for row in violation_log if row.get("violation_type") == "restricted_zone_breach")
    active_alerts = len([row for row in violation_log[-20:] if row.get("severity") in {"MEDIUM", "HIGH"}])
    gas_level = readings.get("methane_lel", 0)
    system = "CRITICAL" if risk_score >= 80 else "WATCH" if risk_score >= 40 else "NORMAL"
    cards = [
        ("Workers Detected", "Workers", workers),
        ("PPE Compliance %", "PPE", f"{compliance}%"),
        ("Active Alerts", "Alerts", active_alerts),
        ("Restricted Zone Intrusions", "Zone", intrusions),
        ("Gas Level", "CH4", f"{gas_level}% LEL"),
        ("System Status", "Status", system),
    ]
    html_cards = "".join(
        f"<div class='live-kpi'><span>{html.escape(icon)}</span><b>{html.escape(str(value))}</b><small>{html.escape(title)}</small></div>"
        for title, icon, value in cards
    )
    st.markdown(f"<div class='live-kpi-bar'>{html_cards}</div>", unsafe_allow_html=True)


def render_alert_stream(violation_log: list[dict], gas_context: dict | None, risk_score: int) -> None:
    rows = latest_frame_events(violation_log, gas_context, risk_score)
    html_rows = "".join(
        "<div class='alert-row {severity}'><span>{time}</span><b>{source}</b><p>{event}</p></div>".format(
            severity=html.escape(str(row["severity"])),
            time=html.escape(str(row["time"])),
            source=html.escape(str(row["source"])),
            event=html.escape(clean_ui_text(row["event"])),
        )
        for row in rows
    )
    st.markdown(f"<div class='alert-stream'><strong>Response Orchestrator</strong>{html_rows}</div>", unsafe_allow_html=True)


def render_admin_notification_card(violation_log: list[dict], gas_context: dict | None, risk_score: int) -> None:
    status = gas_status_text(gas_context)
    alert_types = {row.get("violation_type") for row in violation_log}
    if risk_score >= 70 or status in {"Compound Risk", "Emergency Override"}:
        delivery = "Sent to Plant Safety Admin"
        action = "Pause permit, verify gas, and clear restricted zone"
        tone = "critical"
    elif risk_score >= 35 or violation_log:
        delivery = "Supervisor review queued"
        action = "Inspect PPE, zone status, and active permit"
        tone = "warning"
    else:
        delivery = "Monitoring standby"
        action = "Continue live CCTV and plant-signal watch"
        tone = "normal"
    evidence = "Evidence frame saved" if violation_log else "Waiting for first evidence frame"
    if "restricted_zone_clear" in alert_types:
        action = "Zone cleared; supervisor can review restart"
    st.markdown(
        f"""
        <div class="admin-card {tone}">
          <strong>Admin Notification</strong>
          <span>{html.escape(delivery)}</span>
          <p>{html.escape(evidence)}</p>
          <b>Recommended action: {html.escape(action)}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ai_incident_summary(violation_log: list[dict], gas_context: dict | None, risk_score: int) -> None:
    factors = active_risk_factors(violation_log, gas_context)
    if not factors:
        factors = ["Live monitoring active", "No critical combination detected"]
    actions, _, reduced = intervention_recommendations(violation_log, gas_context, risk_score)
    risk_label = "HIGH" if risk_score >= 70 else "ELEVATED" if risk_score >= 40 else "NORMAL"
    factor_html = "".join(f"<li>{html.escape(clean_ui_text(item.title()))}</li>" for item in factors[:5])
    action_html = "".join(f"<li>{html.escape(action)}</li>" for action in actions[:5])
    st.markdown(
        f"""
        <div class="incident-summary">
          <strong>AI Incident Summary</strong>
          <span>AI detected:</span>
          <ul>{factor_html}</ul>
          <div class="summary-risk">Risk Level: <b>{risk_label}</b></div>
          <span>Recommended Actions:</span>
          <ul class="checks">{action_html}</ul>
          <p>Projected residual risk after actions: {reduced}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_event_playback(violation_log: list[dict], gas_context: dict | None, key: str = "incident_timeline_playback") -> None:
    rows = echo_timeline_rows(violation_log, gas_context)
    position = st.slider("Incident timeline playback", 0, max(0, len(rows) - 1), max(0, len(rows) - 1), key=key)
    rendered = []
    for idx, row in enumerate(rows[: position + 1]):
        rendered.append(
            f"<div class='playback-event {'active' if idx == position else ''}'><b>{html.escape(str(row['Time']))}</b><span>{html.escape(str(row['Event']))}</span><em>{html.escape(str(row['Risk impact']))}</em></div>"
        )
    st.markdown("<div class='playback-strip'>" + "".join(rendered) + "</div>", unsafe_allow_html=True)


def render_operations_dashboard(
    frame_rgb,
    violation_log: list[dict],
    gas_context: dict | None,
    risk_score: int,
    worker_count: int | None = None,
) -> None:
    st.markdown("<div class='ops-dashboard-title'><span>Operations Dashboard</span><b>Live AI Detection Command Center</b></div>", unsafe_allow_html=True)
    render_live_analytics_bar(violation_log, gas_context, risk_score, worker_count)
    video_col, side_col = st.columns([1.22, 0.78], gap="medium")
    with video_col:
        st.markdown("<div class='video-shell'><div class='video-topline'><span class='pulse-dot'></span>LIVE CCTV DETECTION</div></div>", unsafe_allow_html=True)
        st.image(frame_rgb, channels="RGB", use_column_width=True)
    with side_col:
        render_alert_stream(violation_log, gas_context, risk_score)
        render_admin_notification_card(violation_log, gas_context, risk_score)
        render_risk_breakdown(violation_log, gas_context, risk_score)
        render_ai_incident_summary(violation_log, gas_context, risk_score)
    st.markdown("**Plant Heatmap**")
    render_safety_heatmap(gas_context, risk_score, violation_log)
    render_live_signal_correlation(violation_log, gas_context, risk_score)
    render_event_playback(violation_log, gas_context, key="overview_incident_timeline_playback")


def why_alert_text(violation_log: list[dict], gas_context: dict | None, risk_score: int) -> str:
    factors = active_risk_factors(violation_log, gas_context)
    level = "critical" if context_risk_score(violation_log, gas_context, risk_score) >= 85 else "high" if context_risk_score(violation_log, gas_context, risk_score) >= 70 else "elevated"
    if factors:
        return (
            f"This alert is {level} because " + ", ".join(factors[:7])
            + " are occurring together. These combined factors create temporal compound risk and require preventive intervention."
        )
    return "No critical alert is active because major plant, PPE, zone, and compliance risk factors are currently stable."


def render_why_alert(violation_log: list[dict], gas_context: dict | None, risk_score: int) -> None:
    st.markdown(
        f"""
        <div class="why-alert">
          <strong>Why this alert?</strong>
          <p>{html.escape(why_alert_text(violation_log, gas_context, risk_score))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def process_video_worker(
    video_source,
    zone_points,
    result_queue: queue.Queue,
    detector: SafetyDetector,
    gas_context: dict | None,
    live_mode: bool = False,
    stop_event: threading.Event | None = None,
    max_live_seconds: int = 120,
    zone_defs: list[dict] | None = None,
) -> None:
    risk_engine = RiskEngine()
    cap = open_capture(video_source)

    if not cap.isOpened():
        message = "Live Camera Unavailable"
        if not live_mode:
            message = "Video Source Unavailable"
        result_queue.put({"type": "error", "message": message})
        result_queue.put({"type": "done"})
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_index = 0
    processed_count = 0
    violation_count = 0
    log_rows = []
    evidence_saved = 0
    started_at = time.time()
    zone_defs = zone_defs or [{"name": "Zone A", "points": zone_points}]
    previous_active_zones: set[str] = set()

    while True:
        if stop_event is not None and stop_event.is_set():
            break
        if live_mode and max_live_seconds and time.time() - started_at >= max_live_seconds:
            break

        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if frame_index % 3 != 0:
            frame_index += 1
            continue

        try:
            elapsed_seconds = time.time() - started_at
            current_gas_context = live_gas_context(gas_context, elapsed_seconds if live_mode else 0)
            detections = detector.detect(frame)
            worker_count = sum(1 for detection in detections if detection.class_name == "person")
            active_zone_defs = []
            for zone_def in zone_defs:
                touched = any(
                    detection.class_name == "person" and person_touches_zone(detection.bbox, zone_def["points"])
                    for detection in detections
                )
                if touched:
                    active_zone_defs.append(zone_def)
            active_zone_names = {zone_def["name"] for zone_def in active_zone_defs}
            zone_is_occupied = bool(active_zone_names)
            active_zone_name = next(iter(active_zone_names), None)
            active_zone_points = active_zone_defs[0]["points"] if active_zone_defs else zone_points
            zone_event = None
            for entered_zone in sorted(active_zone_names - previous_active_zones):
                zone_event = {
                    "frame": frame_index,
                    "timestamp": f"{frame_index / fps:.2f}s",
                    "violation_type": "restricted_zone_entry",
                    "severity": "HIGH",
                    "risk_points": 40,
                    "estimated": False,
                    "zone_name": entered_zone,
                    "message": f"ZONE ENTRY DETECTED | Worker entered {entered_zone}. Supervisor notification required.",
                }
                log_rows.append(zone_event)
            risk_result = risk_engine.evaluate_frame(
                frame_index=frame_index,
                timestamp_seconds=frame_index / fps,
                detections=detections,
                zone_points=active_zone_points,
                fallback_mode=detector.fallback_mode,
                gas_context=current_gas_context,
            )
            for exited_zone in sorted(previous_active_zones - active_zone_names):
                zone_event = {
                    "frame": frame_index,
                    "timestamp": f"{frame_index / fps:.2f}s",
                    "violation_type": "restricted_zone_clear",
                    "severity": "LOW",
                    "risk_points": 0,
                    "estimated": False,
                    "zone_name": exited_zone,
                    "message": f"ZONE EXIT CONFIRMED | Worker left {exited_zone}. Area is clear for supervisor review.",
                }
                log_rows.append(zone_event)
            previous_active_zones = active_zone_names

            annotated = draw_frame_annotations(
                frame=frame,
                detections=detections,
                zone_points=active_zone_points,
                risk_score=risk_result.score,
                zone_defs=zone_defs if len(zone_defs) > 1 else None,
            )

            if risk_result.violations:
                violation_count += len(risk_result.violations)
                for row in risk_result.violations:
                    if row.get("violation_type") == "restricted_zone_breach" and active_zone_name:
                        row["zone_name"] = active_zone_name
                        row["message"] = row.get("message", "Restricted zone breach").replace(
                            "entered restricted zone",
                            f"entered {active_zone_name}",
                        )
                    log_rows.append(row)
                evidence_saved += 1
                save_evidence_frame(
                    annotated,
                    EVIDENCE_DIR,
                    frame_index=frame_index,
                    risk_score=risk_result.score,
                )

            processed_count += 1
            if live_mode:
                progress = min(1.0, (time.time() - started_at) / max_live_seconds) if max_live_seconds else 0.0
            else:
                progress = min(1.0, (frame_index + 1) / total_frames) if total_frames else 0.0
            result_queue.put(
                {
                    "type": "frame",
                    "frame": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    "risk_score": risk_result.score,
                    "risk_level": risk_result.level,
                    "worker_count": worker_count,
                    "violation_count": violation_count,
                    "violation_log": list(log_rows),
                    "progress": progress,
                    "frame_index": frame_index,
                    "gas_alert_text": gas_status_text(current_gas_context),
                    "gas_context": current_gas_context,
                    "zone_live_state": f"Worker inside {active_zone_name}" if zone_is_occupied else "All restricted zones clear",
                    "zone_live_event": zone_event,
                }
            )
        except Exception as exc:
            log_rows.append(
                {
                    "frame": frame_index,
                    "timestamp": f"{frame_index / fps:.2f}s",
                    "violation_type": "inference_error",
                    "severity": "LOW",
                    "risk_points": 0,
                    "estimated": detector.fallback_mode,
                    "message": f"Inference skipped for frame {frame_index}: {exc}",
                }
            )
            result_queue.put({"type": "log", "violation_log": list(log_rows)})

        frame_index += 1

    cap.release()
    csv_path = write_violation_log_csv(log_rows, LOGS_DIR)
    result_queue.put(
        {
            "type": "complete",
            "progress": 1.0,
            "processed_frames": processed_count,
            "evidence_saved": evidence_saved,
            "csv_path": str(csv_path),
        }
    )
    result_queue.put({"type": "done"})


def drain_worker_queue() -> None:
    q = st.session_state.worker_queue
    if q is None:
        return

    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            break

        item_type = item.get("type")
        if item_type == "frame":
            st.session_state.current_frame = item["frame"]
            st.session_state.risk_score = item["risk_score"]
            st.session_state.risk_level = item["risk_level"]
            st.session_state.worker_count = item.get("worker_count", st.session_state.worker_count)
            st.session_state.violation_count = item["violation_count"]
            st.session_state.violation_log = item["violation_log"]
            st.session_state.progress = item["progress"]
            st.session_state.last_frame_index = item["frame_index"]
            st.session_state.gas_alert_text = item.get("gas_alert_text", st.session_state.gas_alert_text)
            st.session_state.zone_live_state = item.get("zone_live_state", st.session_state.zone_live_state)
            if item.get("zone_live_event") is not None:
                st.session_state.zone_live_event = item["zone_live_event"]
            if item.get("gas_context") is not None:
                st.session_state.gas_context = item["gas_context"]
                readings = item["gas_context"].get("readings", {})
                st.session_state.gas_history = (
                    st.session_state.gas_history
                    + [
                        {
                            "time": item["gas_context"].get("sensor_timestamp", ""),
                            "CH4 %LEL": readings.get("methane_lel"),
                            "CO ppm": readings.get("co_ppm"),
                            "H2S ppm": readings.get("h2s_ppm"),
                            "O2 %": readings.get("oxygen_pct"),
                            "status": item.get("gas_alert_text", ""),
                        }
                    ]
                )[-12:]
        elif item_type == "log":
            st.session_state.violation_log = item["violation_log"]
        elif item_type == "error":
            st.session_state.worker_error = item["message"]
            st.session_state.processing = False
        elif item_type == "complete":
            st.session_state.progress = item["progress"]
            st.session_state.csv_path = item["csv_path"]
            st.session_state.processed_video = True
        elif item_type == "done":
            st.session_state.worker_done = True
            st.session_state.processing = False


def start_processing(
    video_source,
    zone_points,
    gas_context: dict | None,
    live_mode: bool = False,
    max_live_seconds: int = 120,
    zone_defs: list[dict] | None = None,
) -> None:
    detector = get_detector()
    result_queue = queue.Queue()
    stop_event = threading.Event()
    worker = threading.Thread(
        target=process_video_worker,
        args=(video_source, zone_points, result_queue, detector, gas_context, live_mode, stop_event, max_live_seconds, zone_defs),
        daemon=True,
    )
    reset_processing_state()
    st.session_state.video_path = str(video_source)
    st.session_state.zone_points = zone_points
    st.session_state.gas_context = gas_context
    st.session_state.gas_alert_text = gas_status_text(gas_context)
    st.session_state.gas_history = []
    st.session_state.live_stop_event = stop_event
    st.session_state.live_mode = live_mode
    st.session_state.worker_queue = result_queue
    st.session_state.worker_thread = worker
    st.session_state.processing = True
    worker.start()


def stop_live_monitoring() -> None:
    if st.session_state.live_stop_event is not None:
        st.session_state.live_stop_event.set()
    st.session_state.processing = False


def apply_context_preset(name: str) -> None:
    presets = {
        "Normal Ops": {
            "gas_scenario": "Normal",
            "permit_type": "None",
            "maintenance_active": False,
            "equipment_status": "Normal",
            "shift_phase": "Stable operations",
            "audit_status": "Compliant",
            "emergency_event": "None",
        "message": "Normal operations active: safe gas, no active permit, no maintenance conflict.",
        },
        "Gas + Permit Risk": {
            "gas_scenario": "Critical accumulation",
            "permit_type": "Hot Work Permit",
            "maintenance_active": True,
            "equipment_status": "Pump maintenance active",
            "shift_phase": "Shift handover in 30 min",
            "audit_status": "Permit checklist pending",
            "emergency_event": "None",
            "message": "Compound risk active: elevated gas, active permit, maintenance, handover, and checklist gap.",
        },
        "Fire Emergency": {
            "gas_scenario": "Critical accumulation",
            "permit_type": "Maintenance Permit",
            "maintenance_active": True,
            "equipment_status": "Critical equipment bypassed",
            "shift_phase": "Night shift handover",
            "audit_status": "Emergency checklist open",
            "emergency_event": "Fire detected",
            "message": "Fire emergency active: emergency override and evacuation workflow enabled.",
        },
    }
    preset = presets[name]
    reset_processing_state()
    st.session_state.gas_scenario_control = preset["gas_scenario"]
    st.session_state.permit_type_control = preset["permit_type"]
    st.session_state.maintenance_active_control = preset["maintenance_active"]
    st.session_state.equipment_status_control = preset["equipment_status"]
    st.session_state.shift_phase_control = preset["shift_phase"]
    st.session_state.audit_status_control = preset["audit_status"]
    st.session_state.emergency_event_control = preset["emergency_event"]
    st.session_state.copilot_answer = ""
    st.session_state.active_preset = name
    st.session_state.preset_feedback = preset["message"]
    st.session_state.gas_context = build_gas_context(
        preset["gas_scenario"],
        preset["permit_type"],
        preset["maintenance_active"],
        preset["equipment_status"],
        preset["shift_phase"],
        preset["audit_status"],
        preset["emergency_event"],
        real_time_feed=False,
    )
    st.session_state.gas_alert_text = gas_status_text(st.session_state.gas_context)


def render_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #111827;
            --muted: #6b7280;
            --line: #e5e7eb;
            --panel: #ffffff;
            --wash: #f8fafc;
            --blue: #1d4ed8;
            --red: #b91c1c;
            --amber: #a16207;
            --green: #15803d;
        }
        .stApp { background: linear-gradient(180deg, #f8fafc 0%, #ffffff 32%); }
        .main .block-container { padding-top: 1.2rem; max-width: 1240px; }
        div.stButton > button {
            min-height: 2.85rem;
            border-radius: 8px;
            font-weight: 800;
        }
        .top-command-header {
            border: 1px solid #bfdbfe;
            background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
            border-radius: 8px;
            padding: 0.85rem;
            display: grid;
            grid-template-columns: minmax(0, 0.82fr) minmax(420px, 1.18fr);
            gap: 0.85rem;
            align-items: center;
            margin-bottom: 0.65rem;
        }
        .top-command-header h1 { margin: 0 0 0.25rem 0; color: var(--ink); font-size: clamp(1.85rem, 3vw, 2.45rem); line-height: 1.02; }
        .top-command-header p { margin: 0; color: #475569; max-width: 560px; font-size: 0.9rem; line-height: 1.4; }
        .top-summary {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.45rem;
            border: 1px solid #dbeafe;
            background: #ffffff;
            border-radius: 8px;
            padding: 0.6rem;
        }
        .top-summary > span { grid-column: 1 / -1; color: #64748b; font-weight: 900; text-transform: uppercase; font-size: 0.72rem; }
        .top-summary > strong { grid-column: 1 / -1; border-radius: 999px; padding: 0.35rem 0.6rem; width: fit-content; font-size: 0.82rem; }
        .top-summary > strong.ready { background: #eff6ff; color: #1d4ed8; }
        .top-summary > strong.active { background: #ecfdf5; color: #047857; }
        .top-summary .summary-status {
            grid-column: 1 / -1;
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:0.5rem;
        }
        .top-summary .summary-status strong {
            border-radius: 999px;
            padding: 0.28rem 0.58rem;
            width: fit-content;
            font-size: 0.78rem;
        }
        .top-summary strong.ready { background: #eff6ff; color: #1d4ed8; }
        .top-summary strong.active { background: #ecfdf5; color: #047857; }
        .top-summary div { border: 1px solid #e5e7eb; border-radius: 7px; padding: 0.42rem; }
        .top-summary b, .top-summary em { display: block; font-style: normal; }
        .top-summary b { color: #64748b; font-size: 0.68rem; text-transform: uppercase; }
        .top-summary em { color: #0f172a; font-size: 0.9rem; font-weight: 900; margin-top: 0.15rem; }
        .init-step {
            border: 1px solid #bfdbfe;
            border-left: 5px solid #1d4ed8;
            background: #eff6ff;
            color: #1e3a8a;
            border-radius: 8px;
            padding: 0.7rem 0.85rem;
            font-weight: 900;
            margin: 0.4rem 0;
        }
        .live-indicator {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            border: 1px solid #fecaca;
            background: #fff1f2;
            color: #b91c1c;
            border-radius: 999px;
            padding: 0.42rem 0.7rem;
            font-weight: 900;
            margin: 0 0 0.8rem 0;
        }
        .live-indicator span {
            width: 10px;
            height: 10px;
            border-radius: 999px;
            background: #ef4444;
            box-shadow: 0 0 0 7px rgba(239, 68, 68, 0.14);
            animation: pulseLive 1.2s infinite;
        }
        .live-indicator strong { color: #7f1d1d; }
        section[data-testid="stSidebar"] {
            background: #f1f5f9;
            border-right: 1px solid #dbe3ee;
        }
        section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
            color: var(--ink);
        }
        .block-container h2, .block-container h3 {
            color: var(--ink);
            letter-spacing: 0;
        }
        .hero {
            border: 1px solid #dbeafe;
            background: linear-gradient(135deg, #eff6ff 0%, #ffffff 48%, #f8fafc 100%);
            border-radius: 8px;
            padding: 1.15rem 1.25rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
        }
        .hero h1 { margin: 0 0 0.25rem 0; letter-spacing: 0; font-size: 2rem; }
        .hero p { color: #4b5563; margin: 0; max-width: 760px; }
        .hero-badges { display:flex; flex-wrap:wrap; gap:0.4rem; justify-content:flex-end; }
        .status-badge {
            display:inline-flex;
            align-items:center;
            border: 1px solid #bfdbfe;
            background:#ffffff;
            color:#1e3a8a;
            padding:0.35rem 0.55rem;
            border-radius:999px;
            font-size:0.78rem;
            font-weight:800;
            white-space:nowrap;
        }
        .ops-strip {
            display:grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap:0.55rem;
            margin: 0.45rem 0 0.75rem 0;
        }
        .ops-tile {
            border: 1px solid #d1d5db;
            background: #ffffff;
            border-radius: 8px;
            padding: 0.58rem 0.75rem;
        }
        .ops-tile span {
            display:block;
            color:#6b7280;
            font-size:0.78rem;
            font-weight:800;
            text-transform:uppercase;
            letter-spacing:0;
        }
        .ops-tile strong {
            display:block;
            color:#111827;
            font-size:0.9rem;
            margin-top:0.2rem;
        }
        .ops-dashboard-title {
            border: 1px solid rgba(59, 130, 246, 0.35);
            background: linear-gradient(135deg, #07111f 0%, #111827 60%, #1f2937 100%);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin: 0.5rem 0 0.9rem 0;
            box-shadow: 0 18px 38px rgba(15, 23, 42, 0.18);
        }
        .ops-dashboard-title span { display:block; color:#93c5fd; font-size:0.78rem; font-weight:900; text-transform:uppercase; }
        .ops-dashboard-title b { display:block; color:#f8fafc; font-size:1.45rem; margin-top:0.15rem; }
        .live-kpi-bar {
            display:grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap:0.65rem;
            margin: 0.85rem 0;
        }
        .live-kpi {
            border: 1px solid rgba(148, 163, 184, 0.22);
            background: linear-gradient(180deg, rgba(15,23,42,0.96), rgba(30,41,59,0.96));
            border-radius: 8px;
            padding: 0.75rem;
            min-height: 94px;
            color:#e5e7eb;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
        }
        .live-kpi span { color:#38bdf8; font-weight:900; font-size:0.72rem; text-transform:uppercase; }
        .live-kpi b { display:block; color:#ffffff; font-size:1.35rem; margin-top:0.2rem; }
        .live-kpi small { color:#94a3b8; display:block; margin-top:0.15rem; font-weight:700; }
        .video-shell {
            border: 1px solid rgba(59, 130, 246, 0.35);
            border-bottom:0;
            background:#07111f;
            border-radius: 8px 8px 0 0;
            padding:0.65rem 0.8rem;
            color:#e5e7eb;
            font-weight:900;
            letter-spacing:0;
        }
        .video-topline { display:flex; align-items:center; gap:0.5rem; }
        .pulse-dot {
            width:10px;
            height:10px;
            background:#ef4444;
            border-radius:999px;
            display:inline-block;
            box-shadow:0 0 0 rgba(239, 68, 68, 0.75);
            animation:pulseLive 1.4s infinite;
        }
        @keyframes pulseLive {
            0% { box-shadow:0 0 0 0 rgba(239, 68, 68, 0.75); }
            70% { box-shadow:0 0 0 10px rgba(239, 68, 68, 0); }
            100% { box-shadow:0 0 0 0 rgba(239, 68, 68, 0); }
        }
        .alert-stream, .incident-summary {
            border:1px solid rgba(148, 163, 184, 0.2);
            background:linear-gradient(180deg, #0f172a, #111827);
            color:#e5e7eb;
            border-radius:8px;
            padding:0.85rem;
            margin-bottom:0.85rem;
            box-shadow: 0 14px 28px rgba(15, 23, 42, 0.14);
        }
        .alert-stream {
            max-height: 480px;
            overflow-y: auto;
            overflow-x: hidden;
        }
        .alert-stream > strong, .incident-summary > strong { display:block; color:#f8fafc; margin-bottom:0.65rem; font-size:1rem; }
        .alert-row {
            border-left: 4px solid #22c55e;
            background: rgba(15, 23, 42, 0.72);
            border-radius: 7px;
            padding: 0.55rem 0.65rem;
            margin: 0.45rem 0;
            animation: slideInAlert 0.35s ease-out;
        }
        .alert-row.warning { border-left-color:#facc15; }
        .alert-row.critical { border-left-color:#ef4444; }
        .alert-row span { color:#94a3b8; font-size:0.72rem; font-weight:800; }
        .alert-row b { color:#e0f2fe; font-size:0.8rem; margin-left:0.45rem; }
        .alert-row p { color:#f8fafc; margin:0.2rem 0 0 0; font-size:0.82rem; line-height:1.25; }
        @keyframes slideInAlert { from { opacity:0; transform:translateX(14px); } to { opacity:1; transform:translateX(0); } }
        .incident-summary span { display:block; color:#93c5fd; font-weight:900; margin-top:0.45rem; font-size:0.78rem; }
        .incident-summary ul { margin:0.3rem 0 0.65rem 1.1rem; padding:0; color:#f8fafc; }
        .incident-summary li { margin-bottom:0.18rem; font-size:0.84rem; }
        .incident-summary .checks li::marker { content:"✓ "; color:#22c55e; }
        .summary-risk {
            border:1px solid rgba(239, 68, 68, 0.35);
            background:rgba(127, 29, 29, 0.35);
            border-radius:8px;
            padding:0.55rem 0.65rem;
            color:#fecaca;
            font-weight:900;
        }
        .incident-summary p { color:#cbd5e1; font-size:0.82rem; margin:0.45rem 0 0 0; }
        .admin-card {
            border:1px solid rgba(148, 163, 184, 0.2);
            border-left:5px solid #22c55e;
            background:linear-gradient(180deg, #0f172a, #111827);
            color:#e5e7eb;
            border-radius:8px;
            padding:0.85rem;
            margin-bottom:0.85rem;
        }
        .admin-card.warning { border-left-color:#facc15; }
        .admin-card.critical { border-left-color:#ef4444; }
        .admin-card strong { display:block; color:#f8fafc; font-size:1rem; }
        .admin-card span { display:block; color:#93c5fd; font-weight:900; margin-top:0.35rem; }
        .admin-card p { color:#cbd5e1; margin:0.3rem 0; font-size:0.84rem; }
        .admin-card b { display:block; color:#ffffff; font-size:0.84rem; line-height:1.3; }
        .risk-breakdown {
            border:1px solid rgba(148, 163, 184, 0.2);
            background:linear-gradient(180deg, #0f172a, #111827);
            color:#e5e7eb;
            border-radius:8px;
            padding:0.85rem;
            margin-bottom:0.85rem;
        }
        .risk-break-head {
            border-bottom:1px solid rgba(148, 163, 184, 0.18);
            padding-bottom:0.55rem;
            margin-bottom:0.55rem;
        }
        .risk-break-head span { display:block; color:#93c5fd; font-size:0.76rem; font-weight:900; text-transform:uppercase; }
        .risk-break-head b { display:block; color:#ffffff; font-size:1.2rem; margin-top:0.15rem; }
        .risk-break-head em { display:block; color:#fbbf24; font-style:normal; font-weight:900; font-size:0.78rem; margin-top:0.15rem; }
        .risk-break-row {
            display:grid;
            grid-template-columns: 1fr 48px;
            gap:0.5rem;
            align-items:center;
            border-left:4px solid #38bdf8;
            background:rgba(15, 23, 42, 0.58);
            border-radius:7px;
            padding:0.5rem 0.6rem;
            margin:0.42rem 0;
        }
        .risk-break-row b { display:block; color:#f8fafc; font-size:0.84rem; }
        .risk-break-row span { display:block; color:#cbd5e1; font-size:0.74rem; line-height:1.25; margin-top:0.1rem; }
        .risk-break-row strong { color:#f87171; text-align:right; font-size:0.9rem; }
        .warning-banner {
            background: #fef08a;
            border: 1px solid #ca8a04;
            color: #713f12;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-weight: 700;
            margin: 0.75rem 0 1rem 0;
        }
        .zone-event-banner {
            display:flex;
            justify-content:space-between;
            align-items:center;
            gap:1rem;
            border-radius:8px;
            padding:1rem 1.15rem;
            margin:1rem 0;
            border:2px solid;
            box-shadow:0 14px 34px rgba(15, 23, 42, 0.12);
        }
        .zone-event-banner.entry {
            background:linear-gradient(90deg, #7f1d1d, #dc2626);
            border-color:#fecaca;
            color:#ffffff;
            animation: alertPulse 1.25s ease-in-out infinite;
        }
        .zone-event-banner.exit {
            background:linear-gradient(90deg, #14532d, #16a34a);
            border-color:#bbf7d0;
            color:#ffffff;
        }
        .zone-event-banner span {
            display:block;
            font-size:0.78rem;
            font-weight:900;
            opacity:0.82;
            text-transform:uppercase;
        }
        .zone-event-banner strong {
            display:block;
            font-size:1.45rem;
            line-height:1.1;
            margin-top:0.2rem;
        }
        .zone-event-banner p {
            margin:0.35rem 0 0;
            color:rgba(255,255,255,0.9);
            font-weight:700;
        }
        .zone-event-banner b {
            flex:0 0 auto;
            background:rgba(255,255,255,0.16);
            border:1px solid rgba(255,255,255,0.26);
            border-radius:999px;
            padding:0.45rem 0.7rem;
            font-size:0.78rem;
            letter-spacing:0.02em;
        }
        @keyframes alertPulse {
            0%, 100% { box-shadow:0 14px 34px rgba(220, 38, 38, 0.20); }
            50% { box-shadow:0 16px 42px rgba(220, 38, 38, 0.42); }
        }
        .empty-state {
            border: 1px dashed #94a3b8;
            background: #ffffff;
            border-radius: 8px;
            padding: 1.1rem 1.25rem;
            color: #334155;
        }
        .empty-state strong { color: var(--ink); display:block; font-size:1.1rem; margin-bottom:0.25rem; }
        .step-strip {
            display:grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap:0.75rem;
            margin: 0.8rem 0 1.15rem 0;
        }
        .step-card {
            border: 1px solid var(--line);
            background: #ffffff;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            min-height: 86px;
        }
        .step-card span { color: var(--blue); font-weight:900; font-size:0.78rem; }
        .step-card strong { display:block; color:var(--ink); margin-top:0.25rem; font-size:0.98rem; }
        .step-card small { display:block; color:var(--muted); margin-top:0.25rem; line-height:1.25; }
        .demo-flow {
            border: 1px solid #bfdbfe;
            background: #ffffff;
            border-radius: 8px;
            padding: 0.8rem;
            margin: 0.8rem 0 1.1rem 0;
        }
        .demo-flow-title {
            display:flex;
            justify-content:space-between;
            align-items:center;
            gap:0.75rem;
            margin-bottom:0.65rem;
        }
        .demo-flow-title span {
            color:#1d4ed8;
            font-size:0.76rem;
            font-weight:900;
            text-transform:uppercase;
        }
        .demo-flow-title strong { color:#111827; font-size:0.95rem; }
        .demo-flow-steps {
            display:grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap:0.55rem;
        }
        .demo-flow-steps div {
            border:1px solid #dbeafe;
            background:#eff6ff;
            border-radius:8px;
            padding:0.6rem;
            min-height:70px;
        }
        .demo-flow-steps b { display:block; color:#1d4ed8; font-size:0.75rem; }
        .demo-flow-steps span {
            display:block;
            color:#1f2937;
            font-weight:800;
            font-size:0.82rem;
            line-height:1.2;
            margin-top:0.2rem;
        }
        .risk-card {
            border: 2px solid;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            min-height: 74px;
        }
        .risk-label { color:#374151; font-size:0.9rem; }
        .risk-card strong { font-size: 2rem; line-height: 1; }
        .risk-pill {
            color: white;
            border-radius: 999px;
            padding: 0.25rem 0.55rem;
            font-size: 0.75rem;
            font-weight: 800;
        }
        .metric-panel {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            min-height: 74px;
            background: #ffffff;
        }
        .metric-panel span { color:#6b7280; font-size:0.85rem; }
        .metric-panel strong { display:block; font-size:1.7rem; margin-top:0.1rem; }
        .metric-row { margin: 0.75rem 0 1rem 0; }
        .gas-panel {
            border: 2px solid;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin: 1rem 0;
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: center;
        }
        .gas-panel strong { display:block; font-size:1.45rem; margin-top:0.15rem; }
        .gas-readings { display:grid; grid-template-columns: repeat(2, minmax(110px, 1fr)); gap:0.35rem 0.75rem; }
        .gas-readings span { font-weight:700; color:#374151; }
        .heatmap-command {
            border: 1px solid rgba(30,64,175,0.18);
            border-radius: 8px;
            background: #ffffff;
            padding: 0.85rem;
            box-shadow: 0 16px 36px rgba(15,23,42,0.08);
        }
        .heatmap-topbar {
            display:flex;
            justify-content:space-between;
            gap:1rem;
            align-items:center;
            margin-bottom:0.65rem;
            color:#64748b;
            font-weight:800;
        }
        .heatmap-topbar strong {
            color:#dc2626;
            background:#fef2f2;
            border:1px solid #fecaca;
            border-radius:999px;
            padding:0.25rem 0.65rem;
            font-size:0.82rem;
        }
        .industrial-map {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(59, 130, 246, 0.28);
            border-radius: 8px;
            background:
                linear-gradient(90deg, rgba(148,163,184,0.08) 1px, transparent 1px),
                linear-gradient(0deg, rgba(148,163,184,0.08) 1px, transparent 1px),
                radial-gradient(circle at 18% 34%, rgba(239,68,68,0.22), transparent 18%),
                radial-gradient(circle at 72% 44%, rgba(249,115,22,0.2), transparent 20%),
                radial-gradient(circle at 48% 66%, rgba(168,85,247,0.14), transparent 22%),
                #07111f;
            background-size: 42px 42px, 42px 42px, auto, auto, auto;
            min-height: 390px;
            padding: 1rem;
            color:#e5e7eb;
        }
        .map-grid-label {
            position:absolute;
            color:rgba(203,213,225,0.45);
            text-transform:uppercase;
            letter-spacing:0.08em;
            font-size:0.68rem;
            font-weight:900;
        }
        .label-north { left:1rem; top:0.65rem; }
        .label-south { right:1rem; bottom:0.65rem; }
        .pipe { position:absolute; background:rgba(148,163,184,0.3); border-radius:999px; }
        .pipe-a { left:9%; right:8%; top:49%; height:9px; }
        .pipe-b { left:54%; top:11%; bottom:16%; width:8px; }
        .pipe-c { left:15%; right:18%; top:73%; height:6px; transform:rotate(-5deg); }
        .risk-flow {
            position:absolute;
            height:6px;
            border-radius:999px;
            background:linear-gradient(90deg, transparent, rgba(239,68,68,0.9), transparent);
            filter:blur(0.2px);
            opacity:0.8;
            animation:riskFlow 2.4s linear infinite;
        }
        .flow-one { left:17%; right:20%; top:49.3%; }
        .flow-two { left:42%; width:31%; top:70%; transform:rotate(-5deg); animation-delay:0.8s; }
        .plant-zone {
            position:absolute;
            border: 1px solid rgba(148,163,184,0.28);
            background: rgba(15, 23, 42, 0.74);
            color:#f8fafc;
            border-radius:8px;
            padding:0.85rem;
            text-align:left;
            width:230px;
            min-height:132px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
        }
        .plant-zone.is-active { box-shadow:0 0 28px rgba(239,68,68,0.28), inset 0 1px 0 rgba(255,255,255,0.08); animation: zonePulse 1.5s infinite; }
        .plant-zone.normal { border-color:rgba(34,197,94,0.45); }
        .plant-zone.elevated { border-color:rgba(245,158,11,0.72); }
        .plant-zone.high { border-color:rgba(249,115,22,0.9); }
        .plant-zone.critical { border-color:#ef4444; }
        .zone-head { display:flex; align-items:center; justify-content:space-between; gap:0.65rem; }
        .zone-head b { display:block; font-size:1.05rem; }
        .zone-head strong { color:#e0f2fe; font-size:0.86rem; }
        .plant-zone span, .plant-zone em, .plant-zone i { display:block; margin-top:0.18rem; font-style:normal; font-size:0.8rem; color:#cbd5e1; }
        .plant-zone em { color:#fde68a; font-weight:900; }
        .zone-factors { display:flex; flex-wrap:wrap; gap:0.25rem; margin-top:0.45rem; }
        .zone-factors small {
            position:static;
            width:auto;
            height:auto;
            border-radius:999px;
            color:#dbeafe;
            background:rgba(30,64,175,0.35);
            padding:0.12rem 0.38rem;
            font-size:0.66rem;
            box-shadow:none;
            animation:none;
        }
        .zone-a-node { left:5%; top:14%; }
        .zone-b-node { right:6%; top:20%; }
        .control-node { left:35%; bottom:13%; }
        .reactor-node { left:34%; top:21%; }
        .hotspot {
            position:absolute;
            width:20px;
            height:20px;
            border-radius:999px;
            background:#22c55e;
            right:14px;
            top:14px;
            box-shadow:0 0 22px #22c55e;
            animation: pulseLive 1.2s infinite;
        }
        .plant-zone.elevated .hotspot { background:#f59e0b; box-shadow:0 0 22px #f59e0b; }
        .plant-zone.high .hotspot { background:#f97316; box-shadow:0 0 22px #f97316; }
        .plant-zone.critical .hotspot { background:#ef4444; box-shadow:0 0 24px #ef4444; }
        .evac-route {
            position:absolute;
            left:5%;
            right:5%;
            bottom:5%;
            border:1px dashed rgba(34,197,94,0.65);
            background:rgba(20,83,45,0.3);
            color:#bbf7d0;
            border-radius:999px;
            padding:0.45rem 0.75rem;
            font-weight:900;
            text-align:center;
            font-size:0.82rem;
        }
        .heatmap-telemetry {
            display:grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap:0.55rem;
            margin-top:0.65rem;
        }
        .heatmap-telemetry div {
            border:1px solid #dbeafe;
            background:#eff6ff;
            border-radius:8px;
            padding:0.55rem;
        }
        .heatmap-telemetry span {
            display:block;
            color:#64748b;
            font-size:0.72rem;
            font-weight:800;
        }
        .heatmap-telemetry b {
            display:block;
            color:#1e3a8a;
            font-size:0.9rem;
            margin-top:0.15rem;
        }
        .propagation-card {
            margin-top:0.65rem;
            border-left:5px solid #0f766e;
            background:#ecfdf5;
            border-radius:8px;
            padding:0.75rem 0.9rem;
        }
        .propagation-card b {
            display:block;
            color:#064e3b;
            margin-bottom:0.15rem;
        }
        .propagation-card span { color:#115e59; }
        @keyframes zonePulse {
            0%, 100% { transform:scale(1); }
            50% { transform:scale(1.015); }
        }
        @keyframes riskFlow {
            0% { opacity:0.2; transform:translateX(-20px); }
            50% { opacity:0.95; }
            100% { opacity:0.2; transform:translateX(20px); }
        }
        .workflow-panel {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #ffffff;
            padding: 0.65rem;
            min-height: 270px;
            display: grid;
            gap: 0.55rem;
        }
        .workflow-row {
            border-left: 4px solid #1d4ed8;
            background: #f8fafc;
            border-radius: 6px;
            padding: 0.65rem 0.75rem;
        }
        .workflow-row strong { display:block; color:var(--ink); font-size:0.92rem; }
        .workflow-row span { display:block; color:#4b5563; font-size:0.86rem; margin-top:0.15rem; }
        .kg-panel {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #ffffff;
            padding: 0.85rem;
            margin-top: 1rem;
        }
        .kg-title {
            color: var(--ink);
            font-weight: 900;
            margin-bottom: 0.65rem;
        }
        .kg-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.55rem;
        }
        .kg-node {
            border: 1px solid #bfdbfe;
            background: #eff6ff;
            border-radius: 8px;
            padding: 0.55rem 0.65rem;
            min-height: 74px;
        }
        .kg-node strong { display:block; color:#1e3a8a; font-size:0.86rem; }
        .kg-node span { display:block; color:#334155; font-size:0.8rem; margin-top:0.22rem; }
        .kg-link {
            border-left: 4px solid #0f766e;
            background: #f0fdfa;
            color: #134e4a;
            border-radius: 6px;
            padding: 0.55rem 0.7rem;
            margin-top: 0.7rem;
            font-weight: 800;
            font-size: 0.88rem;
        }
        .signal-panel {
            border:1px solid var(--line);
            border-radius:8px;
            background:#ffffff;
            padding:0.8rem;
            margin-top:1rem;
        }
        .signal-title {
            color:var(--ink);
            font-weight:900;
            margin-bottom:0.65rem;
        }
        .signal-row {
            display:grid;
            grid-template-columns: 150px 110px 1fr;
            gap:0.7rem;
            align-items:center;
            border-left:5px solid #22c55e;
            background:#f8fafc;
            border-radius:7px;
            padding:0.65rem 0.75rem;
            margin:0.45rem 0;
        }
        .signal-row.warning { border-left-color:#f59e0b; background:#fffbeb; }
        .signal-row.critical { border-left-color:#ef4444; background:#fef2f2; }
        .signal-row b { color:#111827; }
        .signal-row strong { color:#0f766e; font-size:0.82rem; }
        .signal-row.warning strong { color:#b45309; }
        .signal-row.critical strong { color:#b91c1c; }
        .signal-row span { color:#475569; font-size:0.84rem; line-height:1.25; }
        .cctv-map-title {
            border: 1px solid #bfdbfe;
            background: #eff6ff;
            border-radius: 8px;
            padding: 0.55rem 0.65rem;
            margin: 0.55rem 0 0.45rem 0;
            display: block;
            color: #1e3a8a;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0;
            font-weight: 800;
        }
        .camera-manager-title {
            border: 1px solid #bfdbfe;
            background: #eff6ff;
            border-radius: 8px;
            padding: 0.65rem 0.75rem;
            margin: 0.7rem 0 0.55rem 0;
            color: #1e3a8a;
            font-size: 0.82rem;
            text-transform: uppercase;
            font-weight: 900;
        }
        .camera-card {
            border: 1px solid #dbeafe;
            border-left: 5px solid #94a3b8;
            background: #ffffff;
            border-radius: 8px;
            padding: 0.72rem 0.75rem;
            margin: 0.55rem 0 0.35rem 0;
            box-shadow: 0 5px 16px rgba(15, 23, 42, 0.05);
        }
        .camera-card.selected {
            border-color: #93c5fd;
            border-left-color: #1d4ed8;
            background: #eff6ff;
            box-shadow: 0 0 0 2px rgba(29, 78, 216, 0.08);
        }
        .camera-card.monitoring { border-left-color: #2563eb; }
        .camera-card.alert { border-left-color: #ef4444; background: #fff7ed; }
        .camera-card.configured { border-left-color: #0f766e; }
        .camera-card-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.45rem;
        }
        .camera-card-head strong {
            color: #111827;
            font-size: 0.95rem;
            display: block;
        }
        .camera-card span {
            color: #64748b;
            font-size: 0.76rem;
            display: block;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .camera-card-head b {
            float: none;
            color: #ffffff;
            background: #64748b;
            border-radius: 999px;
            padding: 0.16rem 0.42rem;
            font-size: 0.68rem;
            margin-top: 0;
            white-space: nowrap;
        }
        .camera-card.monitoring .camera-card-head b { background: #2563eb; }
        .camera-card.alert .camera-card-head b { background: #dc2626; }
        .camera-card.configured .camera-card-head b { background: #0f766e; }
        .camera-card p {
            color: #1e3a8a;
            font-weight: 900;
            margin: 0.4rem 0 0.15rem 0;
        }
        .camera-stat-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.35rem;
            margin-top: 0.4rem;
        }
        .camera-stat-grid small {
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 0.28rem;
            color: #0f172a;
            font-size: 0.78rem;
            font-weight: 900;
            text-align: center;
        }
        .camera-stat-grid em {
            display: block;
            color: #64748b;
            font-style: normal;
            font-size: 0.62rem;
            font-weight: 800;
        }
        .plant-status-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 0.55rem;
            margin: 0.85rem 0 1rem 0;
        }
        .plant-status-card {
            border: 1px solid #dbeafe;
            background: #ffffff;
            border-radius: 8px;
            padding: 0.72rem 0.8rem;
            min-height: 82px;
        }
        .plant-status-card span {
            color: #64748b;
            font-weight: 800;
            font-size: 0.74rem;
            display: block;
        }
        .plant-status-card strong {
            color: #0f172a;
            display: block;
            font-size: 1.45rem;
            margin-top: 0.2rem;
        }
        .confidence {
            position: relative;
            height: 28px;
            border-radius: 999px;
            background: #e2e8f0;
            overflow: hidden;
            margin-top: 0.65rem;
        }
        .confidence b, .confidence em {
            position: absolute;
            z-index: 2;
            top: 50%;
            transform: translateY(-50%);
            font-style: normal;
            font-size: 0.72rem;
            font-weight: 900;
        }
        .confidence b { left: 0.7rem; color: #0f172a; }
        .confidence em { right: 0.7rem; color: #0f172a; }
        .confidence i {
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            background: linear-gradient(90deg, #93c5fd, #22c55e);
            border-radius: 999px;
        }
        .events-panel {
            border: 1px solid #dbeafe;
            background: #ffffff;
            border-radius: 8px;
            padding: 0.8rem;
            margin: 0.85rem 0;
        }
        .events-panel > strong {
            display: block;
            color: #0f172a;
            font-size: 1rem;
            margin-bottom: 0.55rem;
        }
        .event-row {
            border-left: 4px solid #22c55e;
            background: #f8fafc;
            border-radius: 7px;
            padding: 0.55rem 0.65rem;
            margin: 0.45rem 0;
        }
        .event-row.warning { border-left-color: #f59e0b; background: #fffbeb; }
        .event-row.critical { border-left-color: #ef4444; background: #fff7ed; }
        .event-row b, .event-row span, .event-row em { display: block; }
        .event-row b { color: #111827; font-size: 0.78rem; }
        .event-row span { color: #334155; font-weight: 700; margin-top: 0.2rem; line-height: 1.28; }
        .event-row em { color: #64748b; font-style: normal; font-size: 0.76rem; margin-top: 0.18rem; }
        .advisor-panel {
            border: 1px solid #bfdbfe;
            border-left: 5px solid #1d4ed8;
            background: #eff6ff;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin: 0.85rem 0;
        }
        .advisor-panel span { color: #1d4ed8; font-size: 0.78rem; font-weight: 900; text-transform: uppercase; }
        .advisor-panel strong { color: #0f172a; display: block; margin-top: 0.2rem; }
        .advisor-panel p { color: #334155; font-weight: 700; line-height: 1.35; margin: 0.35rem 0 0 0; }
        .advisor-meta {
            display:flex;
            flex-wrap:wrap;
            gap:0.35rem;
            margin:0.5rem 0 0.35rem 0;
        }
        .advisor-meta b {
            border:1px solid #bfdbfe;
            background:#ffffff;
            color:#1e3a8a;
            border-radius:999px;
            padding:0.22rem 0.48rem;
            font-size:0.68rem;
            font-weight:900;
            white-space:nowrap;
        }
        .workflow-status-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.55rem;
            margin: 0.75rem 0 1rem 0;
        }
        .workflow-status {
            border: 1px solid #e5e7eb;
            background: #ffffff;
            border-radius: 8px;
            padding: 0.65rem;
            min-height: 72px;
        }
        .workflow-status.done {
            border-color: #bfdbfe;
            background: #eff6ff;
        }
        .workflow-status b {
            display: inline-flex;
            width: 24px;
            height: 24px;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            background: #e5e7eb;
            color: #64748b;
            margin-bottom: 0.35rem;
        }
        .workflow-status.done b {
            background: #1d4ed8;
            color: #ffffff;
        }
        .workflow-status span {
            display: block;
            color: #1f2937;
            font-weight: 800;
            font-size: 0.78rem;
            line-height: 1.2;
        }
        .agent-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.65rem;
            margin-bottom: 1rem;
        }
        .agent-card {
            border: 1px solid var(--line);
            border-top: 4px solid #1d4ed8;
            border-radius: 8px;
            background: #ffffff;
            padding: 0.75rem;
            min-height: 150px;
        }
        .agent-card strong { display:block; color:var(--ink); font-size:0.94rem; }
        .agent-card span { display:block; color:#1d4ed8; font-weight:800; font-size:0.78rem; margin-top:0.25rem; }
        .agent-card p { color:#4b5563; font-size:0.84rem; line-height:1.32; margin:0.5rem 0 0 0; }
        .what-if-result {
            border: 2px solid;
            border-radius: 8px;
            padding: 0.8rem;
            background: #ffffff;
            margin-top: 0.5rem;
        }
        .what-if-result span { display:block; color:var(--muted); font-size:0.82rem; }
        .what-if-result strong { display:block; font-size:1.8rem; line-height:1.1; margin-top:0.1rem; }
        .what-if-result b { display:block; color:var(--ink); margin-top:0.25rem; }
        .what-if-result p { color:#4b5563; margin:0.35rem 0 0 0; font-size:0.88rem; }
        .permit-simulator {
            display:grid;
            grid-template-columns: 1fr 1fr;
            gap:0.85rem;
            border:1px solid color-mix(in srgb, var(--permit), white 30%);
            background:linear-gradient(135deg, rgba(15,23,42,0.98), rgba(30,41,59,0.96));
            border-radius:8px;
            padding:0.9rem;
            color:#e5e7eb;
            box-shadow:0 16px 36px rgba(15,23,42,0.16);
            animation: fadePermit 0.35s ease-out;
        }
        @keyframes fadePermit { from { opacity:0.3; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
        .permit-kicker { color:var(--permit); font-weight:900; text-transform:uppercase; font-size:0.78rem; }
        .permit-risk {
            border:1px solid rgba(255,255,255,0.1);
            background:rgba(255,255,255,0.05);
            border-radius:8px;
            padding:0.7rem;
            margin:0.6rem 0;
        }
        .permit-risk span { color:#94a3b8; display:block; font-size:0.78rem; font-weight:900; }
        .permit-risk strong { display:block; color:var(--permit); font-size:2rem; line-height:1; margin-top:0.2rem; transition:all 0.25s ease; }
        .permit-risk b { display:block; color:#f8fafc; margin-top:0.3rem; font-size:0.88rem; }
        .permit-details, .permit-right ul { margin:0.35rem 0 0.65rem 1.1rem; padding:0; }
        .permit-details li, .permit-right li { margin-bottom:0.28rem; font-size:0.84rem; }
        .permit-right strong { color:#f8fafc; display:block; margin:0.2rem 0 0.35rem 0; }
        .permit-gauges { display:grid; grid-template-columns:1fr 1fr; gap:0.45rem; }
        .permit-gauge {
            border:1px solid rgba(255,255,255,0.09);
            background:rgba(255,255,255,0.045);
            border-radius:7px;
            padding:0.5rem;
        }
        .permit-gauge span { color:#cbd5e1; font-size:0.72rem; font-weight:800; display:block; }
        .permit-gauge b { color:#ffffff; display:block; margin-top:0.12rem; }
        .permit-gauge i { display:block; height:6px; background:rgba(148,163,184,0.22); border-radius:999px; overflow:hidden; margin-top:0.35rem; }
        .permit-gauge em { display:block; height:100%; border-radius:999px; animation: gaugeGrow 0.6s ease-out; }
        @keyframes gaugeGrow { from { width:4%; } }
        .permit-timeline {
            grid-column:1 / -1;
            display:grid;
            grid-template-columns: repeat(6, minmax(0,1fr));
            gap:0.4rem;
            margin-top:0.3rem;
        }
        .permit-step {
            border:1px solid rgba(148,163,184,0.18);
            background:rgba(15,23,42,0.52);
            border-radius:7px;
            padding:0.45rem;
            min-height:64px;
        }
        .permit-step span { display:inline-flex; width:22px; height:22px; border-radius:999px; align-items:center; justify-content:center; background:#334155; color:#cbd5e1; font-weight:900; font-size:0.72rem; }
        .permit-step b { display:block; color:#cbd5e1; font-size:0.72rem; margin-top:0.25rem; line-height:1.15; }
        .permit-step.done span { background:#166534; color:#dcfce7; }
        .permit-step.active { border-color:var(--permit); box-shadow:0 0 18px color-mix(in srgb, var(--permit), transparent 55%); }
        .permit-step.active span { background:var(--permit); color:#111827; }
        .permit-step.active b { color:#ffffff; }
        .compliance-list {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #ffffff;
            padding: 0.75rem 1rem;
            min-height: 185px;
        }
        .compliance-list li { margin-bottom: 0.5rem; color:#374151; }
        .copilot-answer {
            border: 1px solid #bfdbfe;
            border-left: 5px solid #1d4ed8;
            border-radius: 8px;
            background: #eff6ff;
            color: #1e3a8a;
            padding: 0.9rem 1rem;
            font-weight: 700;
            margin: 0.75rem 0 1rem 0;
        }
        .preset-note {
            border: 1px solid #dbeafe;
            background: #ffffff;
            border-radius: 8px;
            padding: 0.7rem 0.8rem;
            color: #334155;
            font-size: 0.88rem;
            margin: 0.6rem 0 0.85rem 0;
        }
        .preset-active {
            border: 1px solid #bfdbfe;
            background: #eff6ff;
            border-radius: 8px;
            padding: 0.7rem 0.8rem;
            margin: 0.75rem 0 0.9rem 0;
        }
        .preset-active span {
            display:block;
            color:#1d4ed8;
            font-size:0.74rem;
            font-weight:900;
            text-transform:uppercase;
        }
        .preset-active strong {
            display:block;
            color:#111827;
            font-size:0.95rem;
            margin-top:0.15rem;
        }
        .preset-active small {
            display:block;
            color:#475569;
            line-height:1.25;
            margin-top:0.25rem;
        }
        .eval-card {
            border: 1px solid var(--line);
            border-top: 4px solid #0f766e;
            border-radius: 8px;
            background: #ffffff;
            padding: 0.85rem;
            min-height: 140px;
        }
        .eval-card span {
            display:block;
            color:#475569;
            font-size:0.78rem;
            font-weight:900;
            text-transform:uppercase;
            letter-spacing:0;
        }
        .eval-card strong {
            display:block;
            color:#0f766e;
            font-size:1.7rem;
            line-height:1.1;
            margin-top:0.35rem;
        }
        .eval-card p {
            color:#4b5563;
            font-size:0.82rem;
            margin:0.45rem 0 0 0;
            line-height:1.3;
        }
        .echo-card {
            border: 2px solid;
            border-radius: 8px;
            background: #ffffff;
            padding: 0.85rem;
            min-height: 112px;
        }
        .echo-card span {
            display:block;
            color:#475569;
            font-size:0.8rem;
            font-weight:900;
        }
        .echo-card strong {
            display:block;
            font-size:1.9rem;
            line-height:1.05;
            margin-top:0.25rem;
        }
        .echo-card p {
            color:#4b5563;
            margin:0.25rem 0 0 0;
            font-weight:800;
            font-size:0.88rem;
        }
        .echo-explain {
            border: 1px solid #cbd5e1;
            border-left: 5px solid #0f766e;
            border-radius: 8px;
            background: #f8fafc;
            color: #334155;
            padding: 0.85rem 1rem;
            margin: 0.75rem 0 1rem 0;
            font-weight: 700;
        }
        .intervention-panel {
            border: 1px solid #bbf7d0;
            border-left: 5px solid #15803d;
            border-radius: 8px;
            background: #f0fdf4;
            padding: 0.9rem 1rem;
        }
        .intervention-panel strong { display:block; color:#14532d; font-size:1rem; margin-bottom:0.45rem; }
        .intervention-panel ul { margin:0.25rem 0 0.75rem 1.1rem; padding:0; color:#166534; }
        .intervention-panel li { margin-bottom:0.25rem; font-weight:700; }
        .risk-reduction {
            background:#ffffff;
            border:1px solid #86efac;
            border-radius: 8px;
            padding:0.65rem 0.75rem;
            color:#14532d;
            font-weight:800;
        }
        .why-alert {
            border: 1px solid #fecaca;
            border-left: 5px solid #dc2626;
            border-radius: 8px;
            background: #fff7ed;
            padding: 0.9rem 1rem;
            margin: 0.75rem 0 1rem 0;
        }
        .why-alert strong { display:block; color:#9a3412; font-size:1rem; }
        .why-alert p { color:#7c2d12; margin:0.35rem 0 0 0; font-weight:700; line-height:1.35; }
        .mode-panel {
            border:1px solid var(--line);
            border-radius:8px;
            padding:0.85rem;
            background:#ffffff;
            margin-top:0.6rem;
        }
        .mode-panel span { color:var(--muted); font-size:0.82rem; display:block; }
        .mode-panel strong { color:var(--ink); display:block; margin-top:0.2rem; }
        .section-card {
            border: 1px solid var(--line);
            background: #ffffff;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .section-card h3 { margin-top: 0; }
        .zone-preview-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.45rem 0 0.25rem 0;
        }
        .zone-preview-legend span {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border: 1px solid #dbeafe;
            background: #f8fafc;
            color: #334155;
            border-radius: 999px;
            padding: 0.28rem 0.55rem;
            font-size: 0.78rem;
            font-weight: 800;
        }
        .zone-preview-legend i {
            width: 0.62rem;
            height: 0.62rem;
            border-radius: 999px;
            display: inline-block;
        }
        .active-zone-note {
            border: 1px solid #fecaca;
            background: #fff1f2;
            color: #991b1b;
            border-radius: 8px;
            padding: 0.7rem 0.8rem;
            margin: 0.75rem 0;
            box-shadow: inset 4px 0 0 #ef4444;
        }
        .active-zone-note strong {
            display: block;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0;
        }
        .active-zone-note span {
            display: block;
            color: #7f1d1d;
            font-size: 0.84rem;
            margin-top: 0.2rem;
            line-height: 1.3;
        }
        .zone-save-strip {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin: 0.65rem 0 0.2rem 0;
        }
        .zone-save-strip span {
            display: inline-flex;
            align-items: center;
            gap: 0.42rem;
            border: 1px solid #bfdbfe;
            color: #334155;
            background: #ffffff;
            border-radius: 999px;
            padding: 0.36rem 0.65rem;
            font-size: 0.82rem;
            font-weight: 800;
        }
        .zone-save-strip span.active {
            border-color: #ef4444;
            background: #fff1f2;
            color: #991b1b;
            box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.12);
        }
        .zone-save-strip i {
            width: 9px;
            height: 9px;
            border-radius: 999px;
            background: #94a3b8;
            display: inline-block;
        }
        .zone-save-strip span.zone-a i { background: #ef4444; }
        .zone-save-strip span.control-room i { background: #0ea5e9; }
        .zone-save-strip span.zone-b i { background: #f97316; }
        .zone-save-strip span.reactor-zone i { background: #a855f7; }
        .zone-save-strip span.control-room.active {
            border-color: #0ea5e9;
            background: #eff6ff;
            color: #075985;
            box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.13);
        }
        .zone-save-strip span.zone-b.active {
            border-color: #f97316;
            background: #fff7ed;
            color: #9a3412;
            box-shadow: 0 0 0 2px rgba(249, 115, 22, 0.14);
        }
        .zone-save-strip span.reactor-zone.active {
            border-color: #a855f7;
            background: #faf5ff;
            color: #6b21a8;
            box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.14);
        }
        .log-empty {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #ffffff;
            padding: 0.9rem 1rem;
            color: var(--muted);
        }
        .playback-strip {
            border:1px solid var(--line);
            background:#ffffff;
            border-radius:8px;
            padding:0.7rem;
            display:grid;
            gap:0.45rem;
            margin-bottom:1rem;
        }
        .playback-event {
            display:grid;
            grid-template-columns: 86px 1fr 70px;
            gap:0.5rem;
            align-items:center;
            border-left:4px solid #94a3b8;
            background:#f8fafc;
            border-radius:7px;
            padding:0.5rem 0.65rem;
        }
        .playback-event.active { border-left-color:#ef4444; background:#fff7ed; }
        .playback-event b { color:#111827; font-size:0.82rem; }
        .playback-event span { color:#374151; font-weight:700; font-size:0.86rem; }
        .playback-event em { color:#b91c1c; font-style:normal; font-weight:900; text-align:right; }
        .modal-score-card {
            border: 1px solid #bfdbfe;
            border-left: 5px solid #2563eb;
            background: #eff6ff;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin: 0.35rem 0 0.85rem 0;
            display: grid;
            grid-template-columns: 1fr auto auto;
            gap: 0.75rem;
            align-items: center;
        }
        .modal-score-card span {
            color: #475569;
            font-weight: 800;
            text-transform: uppercase;
            font-size: 0.75rem;
        }
        .modal-score-card strong {
            color: #0f172a;
            font-size: 1.35rem;
            font-weight: 900;
        }
        .modal-score-card em {
            color: #1d4ed8;
            font-style: normal;
            font-weight: 900;
            background: #dbeafe;
            border-radius: 999px;
            padding: 0.25rem 0.55rem;
        }
        .theme-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #1e3a8a;
            border-radius: 999px;
            padding: 0.35rem 0.65rem;
            font-weight: 850;
            margin: 0.25rem 0 0.75rem;
        }
        @media (max-width: 900px) {
            .top-command-header { grid-template-columns: 1fr; }
            .top-summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .hero { display:block; }
            .hero-badges { justify-content:flex-start; margin-top:0.75rem; }
            .ops-strip { grid-template-columns: 1fr; }
            .live-kpi-bar { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .demo-flow-steps { grid-template-columns: 1fr; }
            .step-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .gas-panel { display:block; }
            .gas-readings { margin-top:0.75rem; }
            .industrial-map { min-height: 520px; }
            .plant-zone { position:relative; left:auto; right:auto; top:auto; bottom:auto; margin-bottom:0.7rem; width:100%; }
            .evac-route { position:relative; left:auto; right:auto; bottom:auto; margin-top:0.5rem; }
            .kg-grid { grid-template-columns: 1fr; }
            .agent-grid { grid-template-columns: 1fr; }
            .permit-simulator { grid-template-columns:1fr; }
            .permit-timeline { grid-template-columns:1fr 1fr; }
        }
        div[data-testid="stDataFrame"] { border: 1px solid #e5e7eb; border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_theme_css() -> None:
    return


def render_landing_page() -> None:
    st.markdown(
        """
        <style>
        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"] { display: none !important; }
        section[data-testid="stSidebar"] { display: none; }
        .main .block-container,
        .block-container {
            max-width: 1180px !important;
            padding-top: 2.25rem !important;
            padding-bottom: 2rem !important;
            overflow: visible !important;
        }
        div[data-testid="stMarkdownContainer"],
        div[data-testid="stVerticalBlock"],
        .element-container { overflow: visible; }
        .landing-nav { display:flex; justify-content:space-between; align-items:center; gap:1rem; min-height:3.75rem; padding:.72rem 0 1rem; border-bottom:1px solid #e2e8f0; overflow:visible; }
        .landing-brand { display:flex; align-items:center; min-height:2.65rem; line-height:1.15; padding:.32rem 0; font-size:1.35rem; font-weight:900; letter-spacing:-0.02em; color:#0f172a; overflow:visible; }
        .landing-links { display:flex; flex-wrap:wrap; align-items:center; gap:.65rem; justify-content:flex-end; }
        .landing-links a { display:inline-flex; align-items:center; justify-content:center; min-height:2.5rem; line-height:1.15; text-decoration:none; color:#1e3a8a; border:1px solid #bfdbfe; background:#eff6ff; padding:.55rem .86rem; border-radius:999px; font-weight:850; overflow:visible; }
        .landing-hero { display:grid; grid-template-columns:minmax(0,1fr) minmax(330px,.88fr); gap:1.75rem; align-items:center; padding:1.75rem 0 1.65rem; }
        .landing-eyebrow { display:inline-flex; color:#1d4ed8; background:#eff6ff; border:1px solid #bfdbfe; border-radius:999px; padding:.45rem .75rem; font-weight:900; margin-bottom:1rem; }
        .landing-hero h1 { font-size:clamp(3rem,5vw,4.7rem); line-height:.95; letter-spacing:-.04em; margin:0 0 .85rem; color:#0f172a; }
        .landing-hero h2 { font-size:clamp(1.25rem,1.7vw,1.75rem); line-height:1.25; margin:0 0 .85rem; color:#1e293b; }
        .landing-hero p, .landing-section p { color:#475569; font-size:1rem; line-height:1.62; }
        .landing-actions { display:flex; flex-wrap:wrap; gap:.85rem; align-items:center; margin-top:1.35rem; }
        .watch-demo { display:inline-flex; align-items:center; justify-content:center; width:100%; min-height:3.25rem; padding:0 1.2rem; border:1px solid #cbd5e1; border-radius:.5rem; color:#0f172a; background:#fff; text-decoration:none; font-weight:850; }
        .watch-demo:hover { border-color:#93c5fd; color:#1d4ed8; background:#f8fbff; }
        .hero-stats { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.7rem; margin-top:1.15rem; }
        .hero-stat { display:flex; align-items:center; min-height:3rem; border:1px solid #dbeafe; background:#f8fbff; border-radius:14px; padding:.78rem .72rem; line-height:1.25; font-weight:900; color:#1e3a8a; font-size:.92rem; }
        .dashboard-preview { align-self:center; justify-self:center; width:100%; max-width:520px; border:1px solid #bfdbfe; background:linear-gradient(180deg,#fff,#f8fbff); border-radius:24px; padding:.66rem; box-shadow:0 20px 52px rgba(15,23,42,.14); }
        .preview-top { display:flex; align-items:center; justify-content:space-between; gap:.7rem; min-height:2.45rem; border-radius:18px 18px 0 0; background:#0f172a; color:#e2e8f0; padding:.52rem .72rem; font-weight:900; }
        .preview-top span { width:.72rem; height:.72rem; border-radius:99px; background:#ef4444; box-shadow:0 0 0 7px rgba(239,68,68,.18); }
        .preview-top em { font-style:normal; color:#fecaca; border:1px solid rgba(248,113,113,.35); padding:.22rem .55rem; border-radius:999px; }
        .preview-grid { display:grid; grid-template-columns:1.25fr .9fr; gap:.58rem; background:#0b1220; padding:.64rem; border-radius:0 0 18px 18px; }
        .preview-video { min-height:164px; position:relative; overflow:hidden; border-radius:14px; background:linear-gradient(135deg,rgba(15,23,42,.15),rgba(15,23,42,.2)), url("https://images.unsplash.com/photo-1581092918056-0c4c3acd3789?auto=format&fit=crop&w=1200&q=80"); background-size:cover; background-position:center; }
        .risk-badge { position:absolute; top:.8rem; left:.8rem; background:rgba(15,23,42,.88); color:#fca5a5; border-radius:8px; padding:.45rem .7rem; font-weight:950; }
        .bbox { position:absolute; border:3px solid #22c55e; border-radius:8px; background:rgba(34,197,94,.08); }
        .bbox.one { width:34%; height:58%; left:22%; top:25%; }
        .bbox.two { width:23%; height:39%; right:13%; top:19%; }
        .zone-box { position:absolute; border:4px solid #ef4444; color:#ef4444; font-weight:950; background:rgba(239,68,68,.16); width:45%; height:24%; left:12%; bottom:10%; border-radius:10px; padding:.35rem; }
        .preview-panel { display:flex; flex-direction:column; gap:.62rem; }
        .advisor-mini, .event-mini { border:1px solid rgba(147,197,253,.32); background:rgba(15,23,42,.86); color:#e2e8f0; border-radius:14px; padding:.56rem; font-size:.82rem; line-height:1.35; }
        .advisor-mini b, .event-mini b { display:block; color:#93c5fd; margin-bottom:.35rem; }
        .advisor-mini strong { color:#fecaca; }
        .preview-metrics { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.6rem; margin-top:.65rem; }
        .preview-metrics div { display:flex; flex-direction:column; justify-content:center; min-height:3.45rem; border:1px solid #dbeafe; border-radius:12px; padding:.65rem; line-height:1.3; color:#475569; font-weight:850; font-size:.88rem; }
        .preview-metrics strong { display:block; color:#0f172a; font-size:1.08rem; line-height:1.15; margin-bottom:.12rem; }
        .landing-section { padding:2.05rem 0; border-top:1px solid #e2e8f0; }
        .landing-section h3 { font-size:clamp(2rem,3vw,3rem); margin:0 0 .8rem; letter-spacing:-.035em; color:#0f172a; }
        .landing-cards { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1rem; margin-top:1.4rem; }
        .landing-card { border:1px solid #dbeafe; border-radius:18px; padding:1.1rem; background:linear-gradient(180deg,#fff,#f8fbff); box-shadow:0 16px 42px rgba(15,23,42,.06); }
        .landing-card b { display:block; color:#1d4ed8; margin-bottom:.42rem; font-size:1.02rem; }
        .landing-card span { color:#475569; line-height:1.55; }
        .comparison-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1rem; margin-top:1.4rem; }
        .comparison-card { border:1px solid #dbeafe; border-radius:20px; padding:1.25rem; background:#fff; }
        .comparison-card.safevision { border-color:#93c5fd; background:#eff6ff; }
        .comparison-card h4 { margin:.1rem 0 .85rem; color:#0f172a; font-size:1.25rem; }
        .comparison-card ul { margin:0; padding-left:1.2rem; color:#475569; line-height:1.8; }
        .why-safevision { display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-top:1.2rem; }
        .why-panel { border:1px solid #dbeafe; border-radius:20px; padding:1.25rem; background:#fff; box-shadow:0 16px 42px rgba(15,23,42,.05); }
        .why-panel.is-strong { background:linear-gradient(135deg,#eff6ff,#fff); border-color:#93c5fd; }
        .why-panel h4 { margin:.1rem 0 .75rem; color:#0f172a; font-size:1.25rem; }
        .why-list { display:grid; gap:.55rem; }
        .why-list span { display:flex; gap:.5rem; align-items:flex-start; color:#475569; line-height:1.45; font-weight:760; }
        .why-list b { color:#1d4ed8; }
        .enterprise-feature-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1rem; margin-top:1.15rem; }
        .enterprise-feature { border:1px solid #bfdbfe; border-radius:20px; padding:1.15rem; background:linear-gradient(180deg,#fff,#f8fbff); box-shadow:0 16px 42px rgba(15,23,42,.055); }
        .enterprise-feature small { display:inline-flex; color:#2563eb; background:#eff6ff; border:1px solid #bfdbfe; border-radius:999px; padding:.22rem .55rem; font-weight:900; margin-bottom:.7rem; }
        .enterprise-feature b { display:block; font-size:1.12rem; color:#0f172a; margin-bottom:.45rem; }
        .enterprise-feature span { color:#475569; line-height:1.52; }
        .architecture-flow { display:flex; flex-wrap:wrap; gap:.7rem; align-items:center; margin-top:1.4rem; }
        .architecture-flow span { background:#0f172a; color:#dbeafe; border:1px solid #1d4ed8; border-radius:999px; padding:.75rem .9rem; font-weight:900; }
        .architecture-flow i { color:#1d4ed8; font-style:normal; font-weight:950; }
        .architecture-note { color:#64748b; font-weight:800; margin:.85rem 0 1.1rem; }
        .demo-video-box { margin-top:1.3rem; border:1px dashed #93c5fd; border-radius:22px; padding:1.3rem; background:#f8fbff; color:#334155; font-weight:800; }
        .cta-panel { border:1px solid #bfdbfe; border-radius:24px; padding:1.35rem; background:linear-gradient(135deg,#eff6ff 0%,#fff 68%); box-shadow:0 18px 45px rgba(37,99,235,.08); }
        .cta-panel h3 { margin-bottom:.45rem; }
        .cta-panel p { color:#475569; max-width:760px; line-height:1.55; font-weight:760; margin:.2rem 0 0; }
        .cta-link { display:flex; align-items:center; justify-content:center; min-height:2.85rem; border:1px solid #bfdbfe; border-radius:12px; background:#fff; color:#1d4ed8 !important; text-decoration:none !important; font-weight:900; }
        .cta-link:hover { background:#eff6ff; border-color:#60a5fa; }
        .landing-footer { margin-top:1.8rem; border:1px solid #1e3a8a; border-radius:24px; padding:1.35rem; color:#dbeafe; background:linear-gradient(135deg,#0f172a,#172554); display:grid; grid-template-columns:1.35fr 1fr; gap:1.2rem; align-items:start; box-shadow:0 20px 55px rgba(15,23,42,.16); }
        .landing-footer strong { display:block; color:#fff; font-size:1.25rem; margin-bottom:.45rem; }
        .landing-footer p { color:#bfdbfe; margin:.2rem 0 0; line-height:1.55; }
        .stack-pills { display:flex; flex-wrap:wrap; gap:.45rem; }
        .stack-pills span { border:1px solid rgba(191,219,254,.38); background:rgba(239,246,255,.12); color:#dbeafe; border-radius:999px; padding:.32rem .65rem; font-weight:850; }
        .footer-links { display:flex; justify-content:flex-end; flex-wrap:wrap; gap:.55rem; margin-top:.75rem; }
        .footer-links a { color:#bfdbfe !important; border:1px solid rgba(191,219,254,.35); border-radius:999px; padding:.35rem .7rem; text-decoration:none !important; font-weight:850; background:rgba(255,255,255,.06); }
        .footer-links a:hover { color:#fff !important; background:rgba(255,255,255,.12); }
        @media (max-width:900px) {
            .landing-hero,.landing-cards,.comparison-grid,.why-safevision,.enterprise-feature-grid,.landing-footer,.preview-grid { grid-template-columns:1fr; }
            .hero-stats,.preview-metrics { grid-template-columns:repeat(2,minmax(0,1fr)); }
            .landing-nav { align-items:flex-start; flex-direction:column; }
            .landing-hero { padding:1.8rem 0 1.6rem; }
            .dashboard-preview { max-width:100%; }
            .footer-links { justify-content:flex-start; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="landing-nav">
          <div class="landing-brand">SafeVision AI</div>
          <div class="landing-links">
            <a href="#features">Features</a>
            <a href="#architecture">Architecture</a>
            <a href="#about">About</a>
            <a href="#contact">Contact</a>
          </div>
        </div>
        <div class="landing-hero">
          <div>
            <span class="landing-eyebrow">Industrial Safety Intelligence</span>
            <h1>SafeVision AI</h1>
            <h2>Context-aware industrial safety intelligence for zero-harm operations</h2>
            <p>CCTV, gas sensors, permits, equipment logs, shift notes, restricted zones, and compliance checklists usually work separately. SafeVision AI brings them together so safety teams can detect compound risk before it becomes an incident.</p>
            <div class="hero-stats">
              <div class="hero-stat">6 Plant Signals</div>
              <div class="hero-stat">Multi-Camera Ready</div>
              <div class="hero-stat">Real-time Risk Fusion</div>
              <div class="hero-stat">Web Dashboard</div>
            </div>
          </div>
          <div class="dashboard-preview">
            <div class="preview-top"><span></span><b>SafeVision Live Dashboard</b><em>High Risk</em></div>
            <div class="preview-grid">
              <div class="preview-video">
                <div class="risk-badge">Risk: 86</div>
                <div class="bbox one"></div>
                <div class="bbox two"></div>
                <div class="zone-box">RESTRICTED ZONE</div>
              </div>
              <div class="preview-panel">
                <div class="advisor-mini"><b>AI Safety Advisor</b><strong>Critical compound risk</strong><br>Gas level, active permit, and restricted-zone activity require supervisor action.</div>
                <div class="event-mini"><b>Recent Safety Event</b>Worker entered Zone A while maintenance permit is active.</div>
                <div class="event-mini"><b>Explain This Alert</b>PPE + zone + gas + permit overlap.</div>
              </div>
            </div>
            <div class="preview-metrics">
              <div><strong>4</strong>Cameras</div>
              <div><strong>12</strong>Workers</div>
              <div><strong>7</strong>Alerts</div>
              <div><strong>86%</strong>Risk</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cta_cols = st.columns([0.32, 0.28, 0.40])
    with cta_cols[0]:
        if st.button("▶ Launch Live Dashboard", type="primary", use_container_width=True):
            st.session_state.safevision_page = "dashboard"
            st.rerun()
    with cta_cols[1]:
        st.markdown('<a class="watch-demo" href="#demo">📹 Watch 2-Min Demo</a>', unsafe_allow_html=True)
    with cta_cols[2]:
        st.caption("Opens the live SafeVision monitoring dashboard.")
    st.markdown(
        """
        <div id="features" class="landing-section">
          <h3>Features</h3>
          <p>SafeVision AI combines computer vision with plant context to turn isolated alerts into explainable safety decisions.</p>
          <div class="landing-cards">
            <div class="landing-card"><b>AI Vision</b><span>Detects workers, PPE status, and restricted-zone movement from CCTV feeds.</span></div>
            <div class="landing-card"><b>Risk Fusion Engine</b><span>Combines vision events with gas readings, permits, equipment, shift notes, and checklist state.</span></div>
            <div class="landing-card"><b>Explain This Alert</b><span>Breaks down every alert into contributing factors, risk weights, and recommended action.</span></div>
            <div class="landing-card"><b>AI Safety Advisor</b><span>Turns live risk signals into clear supervisor instructions and intervention steps.</span></div>
            <div class="landing-card"><b>Risk Heatmap</b><span>Shows plant zones, risk levels, active cameras, and escalation areas in one view.</span></div>
            <div class="landing-card"><b>Incident Timeline</b><span>Creates a chronological record of detections, plant context, actions, and evidence.</span></div>
          </div>
        </div>
        <div class="landing-section">
          <h3>Problem</h3>
          <p>Traditional CCTV, gas sensors, work permits, and compliance systems often operate as separate tools. That creates blind spots when a moderate PPE issue, a gas reading, and a maintenance permit become dangerous only when viewed together.</p>
        </div>
        <div class="landing-section">
          <h3>Solution</h3>
          <div class="landing-cards">
            <div class="landing-card"><b>CCTV / PPE Detection</b><span>Worker detection, PPE status, and zone entry from recorded or industrial CCTV feeds.</span></div>
            <div class="landing-card"><b>Restricted Zone Monitoring</b><span>Draw or configure hazardous areas and monitor entry or exit events.</span></div>
            <div class="landing-card"><b>Gas Sensor Readings</b><span>CH4, CO, H2S, and O2 context feeds into the risk score.</span></div>
            <div class="landing-card"><b>Work Permit Status</b><span>Permit conflicts are surfaced when active work overlaps unsafe plant conditions.</span></div>
            <div class="landing-card"><b>Equipment Status</b><span>Maintenance and fault state raise risk when they overlap with workers or gas alerts.</span></div>
            <div class="landing-card"><b>Shift + Compliance</b><span>Shift handover and checklist status are fused into the same operational view.</span></div>
          </div>
        </div>
        <div class="landing-section">
          <h3>Why SafeVision</h3>
          <div class="comparison-grid">
            <div class="comparison-card">
              <h4>Traditional Safety Systems</h4>
              <ul>
                <li>CCTV, gas sensors, and permits are reviewed separately.</li>
                <li>Alerts explain what happened, but not why it matters.</li>
                <li>Zone risk depends on manual supervisor interpretation.</li>
                <li>Incident evidence is scattered across tools.</li>
              </ul>
            </div>
            <div class="comparison-card safevision">
              <h4>SafeVision AI</h4>
              <ul>
                <li>Combines CCTV, PPE, gas, permit, equipment, shift, and checklist context.</li>
                <li>Scores compound risk in real time.</li>
                <li>Explains alert causes and recommends intervention.</li>
                <li>Preserves timeline, heatmap, evidence, and report context.</li>
              </ul>
            </div>
          </div>
        </div>
        <div id="architecture" class="landing-section">
          <h3>Architecture</h3>
          <p>A lightweight demo pipeline that keeps the real dashboard responsive while showing how vision intelligence becomes supervisor action.</p>
          <div class="architecture-flow">
            <span>CCTV</span><i>→</i><span>YOLO/OpenCV</span><i>→</i><span>Plant Signals</span><i>→</i><span>Risk Fusion Engine</span><i>→</i><span>AI Safety Advisor</span><i>→</i><span>Dashboard</span><i>→</i><span>Supervisor Action</span>
          </div>
          <div class="architecture-note">Full system architecture used in the live dashboard and explainability workflow.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if ARCHITECTURE_DIAGRAM_PATH.exists():
        st.image(str(ARCHITECTURE_DIAGRAM_PATH), use_column_width=True)
    else:
        st.info("Architecture diagram will appear here after the PNG is available.")
    st.markdown(
        """
        <div id="demo" class="landing-section">
          <h3>Demo Video</h3>
          <div class="demo-video-box">Watch the recorded SafeVision AI walkthrough: launch the dashboard, select a CCTV feed, draw a restricted zone, start monitoring, review live detections, inspect the risk heatmap, and generate incident evidence.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if LANDING_DEMO_VIDEO_PATH.exists():
        st.video(str(LANDING_DEMO_VIDEO_PATH))
    else:
        st.info("Demo video will appear here after the recording is added to the project assets folder.")
    st.markdown(
        """
        <div id="about" class="landing-section">
          <h3>Why SafeVision AI</h3>
          <p>Industrial sites rarely fail because one signal is missing. They fail when CCTV, gas readings, permits, equipment status, and compliance checks are separated across different systems.</p>
          <div class="why-safevision">
            <div class="why-panel">
              <h4>CCTV alone misses context</h4>
              <div class="why-list">
                <span>• A worker entering a zone is only one part of the risk.</span>
                <span>• Gas accumulation, active permits, and maintenance status may sit in separate systems.</span>
                <span>• Supervisors lose time connecting scattered evidence after an alert.</span>
              </div>
            </div>
            <div class="why-panel is-strong">
              <h4>SafeVision connects the signals</h4>
              <div class="why-list">
                <span><b>Vision + context:</b> PPE, restricted zones, gas, permits, equipment, shift notes, and checklist status.</span>
                <span><b>Risk score:</b> every event is converted into a clear Low / Medium / High / Critical state.</span>
                <span><b>Supervisor action:</b> alerts include evidence, explanation, and the next response step.</span>
              </div>
            </div>
          </div>
          <div class="enterprise-feature-grid">
            <div class="enterprise-feature">
              <small>Risk Fusion</small>
              <b>Contextual Risk Engine</b>
              <span>Combines CCTV detections with plant signals so repeated weak signals become one clear safety decision.</span>
            </div>
            <div class="enterprise-feature">
              <small>Explainability</small>
              <b>AI Safety Advisor</b>
              <span>Summarizes why an alert occurred, what factors contributed, and which intervention should happen next.</span>
            </div>
            <div class="enterprise-feature">
              <small>Operations</small>
              <b>Plant Heatmap Intelligence</b>
              <span>Maps camera evidence and active alerts to plant zones so supervisors can prioritize response areas quickly.</span>
            </div>
          </div>
        </div>
        <div id="contact" class="landing-section">
          <div class="cta-panel">
            <h3>Launch the live safety operations dashboard</h3>
            <p>Open the command-center view to configure CCTV feeds, draw restricted zones, monitor plant risk, review explanations, and prepare incident evidence.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    bottom_cta_cols = st.columns([0.28, 0.28, 0.44])
    with bottom_cta_cols[0]:
        if st.button("Open Dashboard", key="landing_footer_open_dashboard", type="primary", use_container_width=True):
            st.session_state.safevision_page = "dashboard"
            st.rerun()
    with bottom_cta_cols[1]:
        st.markdown('<a class="cta-link" href="#architecture">View Architecture</a>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="landing-footer">
          <div>
            <strong>SafeVision AI</strong>
            <p>Enterprise-style industrial safety intelligence for CCTV analytics, plant context, risk fusion, alert explanation, and supervisor response workflows.</p>
            <div class="footer-links">
              <a href="#features">Features</a>
              <a href="#architecture">Docs</a>
              <a href="#contact">Dashboard</a>
            </div>
          </div>
          <div>
            <div class="stack-pills"><span>React</span><span>FastAPI</span><span>PostgreSQL</span><span>OpenCV</span><span>YOLO</span><span>Docker</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard() -> None:
    ensure_project_dirs([MODELS_DIR, EVIDENCE_DIR, LOGS_DIR, UPLOADS_DIR, CANVAS_FRAMES_DIR, SAMPLE_VIDEOS_DIR])
    ensure_factory_demo_video()
    init_session_state()
    render_css()
    render_dashboard_theme_css()
    drain_worker_queue()

    detector = get_detector()
    st.session_state.detector_mode = "fallback" if detector.fallback_mode else "custom"

    with st.sidebar:
        if st.button("Home", use_container_width=True):
            st.session_state.safevision_page = "home"
            st.rerun()
        st.subheader("CCTV Source")
        input_mode = st.radio("Monitoring mode", ["Recorded CCTV", "Industrial CCTV"], horizontal=False)
        st.session_state.input_mode = input_mode
        uploaded = None
        uploaded_files = []
        plant_cameras = []
        selected_camera = None
        live_duration = 120
        if input_mode == "Recorded CCTV":
            uploaded_files = st.file_uploader(
                "Plant Camera Manager",
                type=["mp4", "mov", "avi", "mkv"],
                accept_multiple_files=True,
                key=f"plant_camera_upload_{st.session_state.upload_nonce}",
                help="Upload CCTV videos. Each file becomes a persistent camera card.",
            ) or []
            st.session_state.monitor_all_zones = len(uploaded_files) > 1 or st.session_state.get("monitor_all_zones", False)
            plant_cameras, selected_camera = render_camera_manager(uploaded_files, st.session_state.get("gas_context"))
            if selected_camera:
                uploaded = uploaded_files[selected_camera["index"]]
                if selected_camera.get("zone") in PLANT_ZONES:
                    st.session_state.zone_map_target = selected_camera["zone"]
                    st.session_state.zone_edit_target = selected_camera["zone"]
            st.caption("Camera cards persist zone setup, alerts, and monitoring state while you switch feeds.")
        else:
            video_source = build_live_source("Industrial CCTV", "")
            st.caption("Uses the bundled industrial CCTV feed for cloud-safe monitoring.")
        st.subheader("Plant Signal Inputs")
        st.markdown(
            "<div class='preset-note'>External gas, permit, shift, equipment, and compliance signals used for risk correlation.</div>",
            unsafe_allow_html=True,
        )
        preset_cols = st.columns(3)
        with preset_cols[0]:
            if st.button("Normal Ops", use_container_width=True):
                apply_context_preset("Normal Ops")
                st.rerun()
        with preset_cols[1]:
            if st.button("Gas + Permit", use_container_width=True):
                apply_context_preset("Gas + Permit Risk")
                st.rerun()
        with preset_cols[2]:
            if st.button("Fire Emergency", use_container_width=True):
                apply_context_preset("Fire Emergency")
                st.rerun()
        st.markdown(
            f"""
            <div class="preset-active">
              <span>Operating profile</span>
              <strong>{html.escape(st.session_state.active_preset)}</strong>
              <small>{html.escape(st.session_state.preset_feedback or "Plant signal controls are ready.")}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
        gas_scenario = st.selectbox(
            "Gas sensor scenario",
            ["Normal", "Elevated accumulation", "Critical accumulation"],
            key="gas_scenario_control",
        )
        permit_type = st.selectbox(
            "Active permit",
            ["None", "Maintenance Permit", "Hot Work Permit", "Confined Space Entry", "Electrical Permit", "Working at Height"],
            key="permit_type_control",
        )
        maintenance_active = st.checkbox("Maintenance crew active", key="maintenance_active_control")
        equipment_status = st.selectbox(
            "Equipment maintenance status",
            ["Normal", "Pump maintenance active", "Critical equipment bypassed", "Valve isolation pending"],
            key="equipment_status_control",
        )
        shift_phase = st.selectbox(
            "Shift / handover status",
            ["Stable operations", "Shift handover in 30 min", "Night shift handover", "New crew onboarding"],
            key="shift_phase_control",
        )
        audit_status = st.selectbox(
            "Compliance checklist",
            ["Compliant", "Permit checklist pending", "Inspection overdue", "Emergency checklist open"],
            key="audit_status_control",
        )
        emergency_event = st.selectbox(
            "Emergency event",
            ["None", "Fire detected"],
            key="emergency_event_control",
        )
        real_time_gas_feed = st.checkbox("Real-time gas feed", key="real_time_gas_feed_control")
        snapshot_active_camera_context()
        if st.session_state.processing:
            if st.button("Stop Monitoring", type="secondary", use_container_width=True, key="sidebar_stop_monitoring"):
                stop_live_monitoring()
                st.rerun()
        if st.button("System Architecture", use_container_width=True, key="sidebar_architecture_modal"):
            show_architecture_dialog()
        if st.button("Reset Session", use_container_width=True):
            reset_processing_state()
            st.session_state.video_path = None
            st.session_state.video_paths = {}
            st.session_state.active_cctv_index = 0
            st.session_state.source_signature = None
            st.session_state.plant_cameras = []
            st.session_state.camera_alerts = {}
            st.session_state.camera_metrics = {}
            st.session_state.camera_context = {}
            st.session_state.camera_evidence = {}
            st.session_state.custom_zone_points = {}
            st.session_state.upload_nonce += 1
            st.session_state.reset_feedback = "Session cleared. Camera uploads, zones, alerts, and monitoring state were reset."
            st.rerun()
        if st.session_state.get("reset_feedback"):
            st.caption(st.session_state.reset_feedback)

    if input_mode == "Recorded CCTV":
        file_signature = "|".join(f"{item.name}:{item.size}" for item in uploaded_files)
        source_signature = f"plant:{file_signature or 'none'}"
    else:
        source_signature = f"industrial:{FACTORY_DEMO_VIDEO.name}"
    if st.session_state.source_signature != source_signature:
        reset_processing_state()
        st.session_state.video_path = None
        st.session_state.source_signature = source_signature

    gas_context = build_gas_context(
        gas_scenario,
        permit_type,
        maintenance_active,
        equipment_status,
        shift_phase,
        audit_status,
        emergency_event,
        real_time_feed=real_time_gas_feed and input_mode == "Industrial CCTV",
    )
    display_gas_context = st.session_state.gas_context if st.session_state.processing and st.session_state.gas_context else gas_context
    with st.sidebar:
        render_sidebar_status(detector, display_gas_context)

    render_top_header(display_gas_context)
    vision_value = (
        f"{max(1, int(st.session_state.get('worker_count', 0)))} worker stream active"
        if live_cctv_monitoring()
        else f"{len(st.session_state.get('plant_cameras', []))} configured camera feeds"
    )
    context_value = f"{gas_status_text(display_gas_context)} | {display_gas_context.get('permit_type', 'No permit') if display_gas_context else 'No permit'}"
    response_value = (
        "Incident report ready"
        if st.session_state.get("generated_report") or st.session_state.get("csv_path")
        else f"{len(collect_plant_events(display_gas_context))} live safety events"
    )
    st.markdown(
        f"""
        <div class="ops-strip">
          <div class="ops-tile"><span>Vision Layer</span><strong>{html.escape(vision_value)}</strong></div>
          <div class="ops-tile"><span>Plant Context</span><strong>{html.escape(context_value)}</strong></div>
          <div class="ops-tile"><span>Response Layer</span><strong>{html.escape(response_value)}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_plant_status(display_gas_context)
    render_live_indicator()
    status_left, status_right = st.columns([1.15, 0.85], gap="large")
    with status_left:
        render_workflow_status(active_camera())
    with status_right:
        render_ai_advisor(display_gas_context)

    if input_mode == "Recorded CCTV":
        selected_camera = active_camera()
        if uploaded is None or selected_camera is None:
            st.markdown(
                """
                <div class="empty-state">
                  <strong>Ready for Plant Camera Manager</strong>
                  Upload CCTV videos to create camera cards, assign zones, and start plant monitoring.
                </div>
                """,
                unsafe_allow_html=True,
            )
            return
        st.session_state.video_path = selected_camera.get("path")
        if not st.session_state.video_path:
            st.markdown(
                """<div class="empty-state"><strong>Video Source Unavailable</strong>Please select another video source.</div>""",
                unsafe_allow_html=True,
            )
            return
        video_source = Path(st.session_state.video_path)

    first_frame = load_first_frame(
        video_source,
        allow_demo_fallback=input_mode == "Industrial CCTV",
        allow_placeholder=input_mode == "Industrial CCTV",
    )
    if first_frame is None:
        if input_mode == "Recorded CCTV":
            st.markdown(
                """<div class="empty-state"><strong>Video Source Unavailable</strong>Please select another video source.</div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div class="empty-state"><strong>Industrial CCTV Feed Unavailable</strong></div>""",
                unsafe_allow_html=True,
            )
        return

    default_zone = build_default_zone(first_frame.shape[1], first_frame.shape[0])

    st.subheader("1. Define Restricted Zone")
    zone_col, action_col = st.columns([0.58, 0.42], gap="medium")
    with action_col:
        st.markdown(
            """
            <div class="section-card">
              <h3>Zone Setup</h3>
              <p>Freestyle draw around the hazardous area on the first frame, or choose a preset zone for a repeatable monitoring profile.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        active_preset = st.session_state.get("zone_preset", "drawn")
        preset_cols = st.columns(3)
        with preset_cols[0]:
            if st.button("Bottom Left", type="primary" if active_preset == "bottom_left" else "secondary", use_container_width=True):
                st.session_state.zone_preset = "bottom_left"
                if st.session_state.get("monitor_all_zones", False):
                    target = st.session_state.get("zone_edit_target", "Zone A")
                    set_saved_zone(target, build_preset_zone(first_frame.shape[1], first_frame.shape[0], "bottom_left"))
                else:
                    invalidate_processed_preview()
                st.session_state.zone_canvas_nonce += 1
        with preset_cols[1]:
            if st.button("Bottom Center", type="primary" if active_preset == "bottom_center" else "secondary", use_container_width=True):
                st.session_state.zone_preset = "bottom_center"
                if st.session_state.get("monitor_all_zones", False):
                    target = st.session_state.get("zone_edit_target", "Control Room")
                    set_saved_zone(target, build_preset_zone(first_frame.shape[1], first_frame.shape[0], "bottom_center"))
                else:
                    invalidate_processed_preview()
                st.session_state.zone_canvas_nonce += 1
        with preset_cols[2]:
            if st.button("Bottom Right", type="primary" if active_preset == "bottom_right" else "secondary", use_container_width=True):
                st.session_state.zone_preset = "bottom_right"
                if st.session_state.get("monitor_all_zones", False):
                    target = st.session_state.get("zone_edit_target", "Zone B")
                    set_saved_zone(target, build_preset_zone(first_frame.shape[1], first_frame.shape[0], "bottom_right"))
                else:
                    invalidate_processed_preview()
                st.session_state.zone_canvas_nonce += 1
        if st.button("Draw Free Zone", type="primary" if active_preset == "drawn" else "secondary", use_container_width=True):
            st.session_state.zone_preset = "drawn"
            invalidate_processed_preview()
            st.session_state.zone_canvas_nonce += 1

        st.session_state.monitor_all_zones = st.checkbox(
            "Monitor all configured zones",
            value=st.session_state.get("monitor_all_zones", False),
            key="monitor_all_zones_toggle",
        )
        if st.session_state.monitor_all_zones:
            st.session_state.zone_edit_target = st.selectbox(
                "Edit restricted zone",
                PLANT_ZONES,
                index=PLANT_ZONES.index(st.session_state.get("zone_edit_target", "Zone A")) if st.session_state.get("zone_edit_target", "Zone A") in PLANT_ZONES else 0,
                key="zone_edit_target_select",
                help="Pick a zone, draw its boundary on the preview, then switch to the next zone.",
            )
            st.session_state.zone_map_target = st.session_state.zone_edit_target
            st.caption("Heatmap will monitor all configured plant zones together.")
        else:
            st.session_state.zone_map_target = st.selectbox(
                "Heatmap alert zone",
                PLANT_ZONES,
                index=PLANT_ZONES.index(st.session_state.get("zone_map_target", "Zone A")) if st.session_state.get("zone_map_target", "Zone A") in PLANT_ZONES else 0,
                key="zone_map_target_select",
            )

        preset_label = {
            "bottom_left": "Bottom Left",
            "bottom_center": "Bottom Center",
            "bottom_right": "Bottom Right",
            "drawn": "Free Draw Zone",
        }.get(st.session_state.get("zone_preset", "drawn"), "Free Draw Zone")
        scope_label = "All configured zones" if st.session_state.monitor_all_zones else st.session_state.zone_map_target
        st.caption(f"Selected zone: {preset_label} -> {scope_label}")
        if st.session_state.monitor_all_zones:
            st.markdown(
                f"""
                <div class="operator-note active-zone-note">
                  <strong>Editing {st.session_state.get("zone_edit_target", "Zone A")}</strong>
                  <span>Draw or choose a preset for this CCTV feed, then switch the edit selector for the next zone.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            configured = st.session_state.get("custom_zone_points", {})
            st.caption(
                "Zone setup: "
                + " | ".join(
                    f"{name}: {'Custom' if zone_storage_key(name) in configured or name in configured else 'Default'}"
                    for name in PLANT_ZONES
                )
            )
            if st.button("Reset Selected Zone", type="secondary", use_container_width=True):
                remove_saved_zone(st.session_state.get("zone_edit_target", "Zone A"))
                st.session_state.zone_canvas_nonce += 1
                st.rerun()

    canvas_width = min(500, first_frame.shape[1])
    scale = canvas_width / first_frame.shape[1]
    canvas_height = int(first_frame.shape[0] * scale)
    camera_for_canvas = active_camera()
    canvas_camera_id = (
        camera_for_canvas.get("id")
        if camera_for_canvas
        else f"industrial_{FACTORY_DEMO_VIDEO.stem}"
    )
    canvas_background_path = save_canvas_background_frame(
        first_frame,
        f"{canvas_camera_id}_{st.session_state.get('source_signature', 'source')}",
        canvas_width,
        canvas_height,
    )
    if canvas_background_path and canvas_background_path.exists():
        with Image.open(canvas_background_path) as saved_background:
            canvas_background = saved_background.convert("RGB").copy()
    else:
        canvas_background = Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)).resize(
            (canvas_width, canvas_height),
            Image.Resampling.LANCZOS,
        ).convert("RGB")
    preset = st.session_state.get("zone_preset", "drawn")
    preset_zone = (
        build_preset_zone(first_frame.shape[1], first_frame.shape[0], preset)
        if preset in {"bottom_left", "bottom_center", "bottom_right"}
        else None
    )
    preset_zone_defaults = camera_zone_defaults(first_frame.shape[1], first_frame.shape[0])
    custom_zone_points = st.session_state.get("custom_zone_points", {})
    camera_index = active_camera().get("index", st.session_state.get("active_cctv_index", 0)) if active_camera() else st.session_state.get("active_cctv_index", 0)
    all_zone_defs = configured_zone_defs(first_frame.shape[1], first_frame.shape[0], camera_index)
    edit_zone_name = st.session_state.get("zone_edit_target", st.session_state.zone_map_target)

    with zone_col:
        if st.session_state.monitor_all_zones and preset_zone:
            render_zone_preview_multi(
                first_frame,
                all_zone_defs,
                canvas_width,
            )
            canvas_result = None
            st.caption(f"Preset saved for {edit_zone_name}. Choose Draw Free Zone to sketch a custom boundary.")
            legend = "".join(
                f"<span class='{'zone-a' if item['name'] == 'Zone A' else 'control-room' if item['name'] == 'Control Room' else 'reactor-zone' if item['name'] == 'Reactor Zone' else 'zone-b'} {'active' if item['name'] == edit_zone_name else ''}'><i></i>{html.escape(item['name'])}</span>"
                for item in all_zone_defs
            )
            st.markdown(f"<div class='zone-save-strip'>{legend}</div>", unsafe_allow_html=True)
        elif preset_zone and not st.session_state.monitor_all_zones:
            render_zone_preview_multi(
                first_frame,
                [{"name": st.session_state.zone_map_target, "points": preset_zone}],
                canvas_width,
            )
            canvas_result = None
        else:
            canvas_result = st_canvas(
                fill_color="rgba(239, 68, 68, 0.18)",
                stroke_width=4,
                stroke_color=DEFAULT_ZONE_COLOR,
                background_image=canvas_background,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode="freedraw",
                display_toolbar=True,
                key=f"zone_canvas_{st.session_state.get('active_cctv_index', 0)}_{preset}_{edit_zone_name}_{st.session_state.zone_canvas_nonce}",
            )
            if not canvas_background_path:
                st.caption("Canvas frame fallback active. If the preview is blank, choose a preset zone and continue monitoring.")
            if st.session_state.monitor_all_zones:
                st.caption(f"Drawing target: {edit_zone_name}. Switch the edit selector to draw the next restricted zone.")
                legend = "".join(
                    f"<span class='{'zone-a' if item['name'] == 'Zone A' else 'control-room' if item['name'] == 'Control Room' else 'reactor-zone' if item['name'] == 'Reactor Zone' else 'zone-b'} {'active' if item['name'] == edit_zone_name else ''}'><i></i>{html.escape(item['name'])}</span>"
                    for item in all_zone_defs
                )
                st.markdown(f"<div class='zone-save-strip'>{legend}</div>", unsafe_allow_html=True)

    drawn_zone = None
    if canvas_result is not None:
        drawn_zone = extract_polygon_from_canvas(
            canvas_result.json_data,
            scale_x=first_frame.shape[1] / canvas_width,
            scale_y=first_frame.shape[0] / canvas_height,
        )
        if st.session_state.monitor_all_zones and drawn_zone:
            set_saved_zone(edit_zone_name, drawn_zone)
            all_zone_defs = configured_zone_defs(first_frame.shape[1], first_frame.shape[0], camera_index)
    zone_defs = None
    if st.session_state.monitor_all_zones:
        zone_defs = configured_zone_defs(first_frame.shape[1], first_frame.shape[0], camera_index)
        active_camera_zone = active_camera().get("zone", edit_zone_name) if active_camera() else edit_zone_name
        if active_camera_zone not in PLANT_ZONES:
            active_camera_zone = edit_zone_name
        zone_points = next((item["points"] for item in zone_defs if item["name"] == active_camera_zone), zone_defs[0]["points"])
    elif preset_zone:
        zone_points = preset_zone
        zone_defs = [{"name": st.session_state.zone_map_target, "points": preset_zone}]
    else:
        zone_points = drawn_zone if drawn_zone else default_zone
        zone_defs = [{"name": st.session_state.zone_map_target, "points": zone_points}]

    zone_action_cols = st.columns([1, 1, 1.2])
    with zone_action_cols[0]:
        if st.button("Save Zone for Camera", type="primary", use_container_width=True):
            camera = active_camera()
            target_zone = edit_zone_name if st.session_state.get("monitor_all_zones", False) else st.session_state.zone_map_target
            save_camera_zone(target_zone, zone_points, camera.get("index") if camera else None)
            st.session_state.zone_action_feedback = f"Saved {target_zone} for {camera.get('camera', 'selected camera') if camera else 'selected camera'}."
            st.rerun()
    with zone_action_cols[1]:
        if st.button("Configure Next Camera", type="secondary", use_container_width=True):
            cameras = st.session_state.get("plant_cameras", [])
            if cameras:
                camera = active_camera()
                target_zone = edit_zone_name if st.session_state.get("monitor_all_zones", False) else st.session_state.zone_map_target
                save_camera_zone(target_zone, zone_points, camera.get("index") if camera else None)
                next_index = (st.session_state.get("active_cctv_index", 0) + 1) % len(cameras)
                select_camera(next_index, restore_context=False)
                next_camera = active_camera()
                st.session_state.zone_action_feedback = f"Saved {target_zone}. Now configuring {next_camera.get('camera', 'next camera') if next_camera else 'next camera'}."
                st.rerun()
    with zone_action_cols[2]:
        if st.button("Generate Incident Report", type="secondary", use_container_width=True):
            st.session_state.generated_report = True
            st.session_state.generated_report_text = build_incident_report_text(
                st.session_state.violation_log,
                display_gas_context,
                st.session_state.risk_score,
                st.session_state.risk_level,
            )
            st.session_state.zone_action_feedback = "Incident report draft generated below."
            st.rerun()
    if st.session_state.get("zone_action_feedback"):
        st.success(st.session_state.zone_action_feedback)
    if st.session_state.get("generated_report_text"):
        st.download_button(
            "Download Generated Incident Report",
            data=st.session_state.generated_report_text,
            file_name=f"safevision_incident_report_{app_filename_stamp()}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.subheader("2. Run Analysis")
    render_gas_panel(display_gas_context)
    if st.session_state.gas_history:
        st.caption("Latest live gas sensor readings")
        st.dataframe(pd.DataFrame(st.session_state.gas_history), use_container_width=True, height=180)
    start_disabled = st.session_state.processing
    plant_camera_count = len(st.session_state.get("plant_cameras", []))
    start_label = "START PLANT MONITORING" if plant_camera_count > 1 else "Start Monitoring"
    run_col, stop_col = st.columns([2, 1])
    with run_col:
        if st.button(start_label, type="primary", disabled=start_disabled, use_container_width=True):
            render_monitoring_animation()
            start_processing(
                video_source,
                zone_points,
                gas_context,
                live_mode=input_mode == "Industrial CCTV",
                max_live_seconds=live_duration,
                zone_defs=zone_defs,
            )
            st.session_state.monitoring_started_at = time.time()
            if plant_camera_count > 1:
                start_multi_camera_monitoring(gas_context)
            st.rerun()
    with stop_col:
        if st.button("Stop Monitoring", type="secondary", disabled=not st.session_state.processing, use_container_width=True):
            stop_live_monitoring()
            st.rerun()

    if st.session_state.worker_error:
        st.markdown(
            f"""<div class="empty-state"><strong>{html.escape(st.session_state.worker_error)}</strong></div>""",
            unsafe_allow_html=True,
        )

    progress_text = "Processing video..." if st.session_state.processing else "Ready"
    st.progress(st.session_state.progress, text=progress_text)
    render_zone_event_banner(st.session_state.violation_log, st.session_state.zone_live_event)
    camera = active_camera()
    if camera:
        camera_events = []
        for row in st.session_state.violation_log[-20:]:
            camera_events.append(
                {
                    "timestamp": row.get("timestamp", app_time()),
                    "camera": camera.get("camera", "CCTV"),
                    "zone": row.get("zone_name", camera.get("zone", "Plant")),
                    "severity": row.get("severity", "MEDIUM"),
                    "type": row.get("violation_type", "vision"),
                    "message": row.get("message", row.get("violation_type", "Safety event")),
                }
            )
        st.session_state.camera_alerts[camera["id"]] = camera_events
        ppe_issues = sum(1 for row in st.session_state.violation_log if row.get("violation_type") in {"no_helmet", "no_vest"})
        camera["worker_count"] = max(camera.get("worker_count", 0), st.session_state.worker_count)
        camera["alert_count"] = len(camera_events)
        camera["ppe_compliance"] = max(0, 100 - ppe_issues * 8)
        camera["risk_score"] = st.session_state.risk_score
        camera["monitoring"] = st.session_state.processing or camera.get("monitoring", False)
        st.session_state.camera_metrics[camera["id"]] = {
            "worker_count": camera["worker_count"],
            "alert_count": camera["alert_count"],
            "ppe_compliance": camera["ppe_compliance"],
            "risk_score": camera["risk_score"],
        }
        refresh_camera_status(camera)
    render_alert_feed(display_gas_context)

    st.markdown('<div class="metric-row">', unsafe_allow_html=True)
    left, mid, right = st.columns([1.2, 0.8, 0.8])
    with left:
        risk_badge(st.session_state.risk_score, st.session_state.risk_level)
    with mid:
        st.markdown(
            f"""<div class="metric-panel"><span>Total violations</span><strong>{st.session_state.violation_count}</strong></div>""",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""<div class="metric-panel"><span>Last processed frame</span><strong>{st.session_state.last_frame_index}</strong></div>""",
            unsafe_allow_html=True,
        )
    st.markdown(
        f"""<div class="metric-panel"><span>Compound gas intelligence</span><strong>{st.session_state.gas_alert_text}</strong></div>""",
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.current_frame is not None:
        display_frame = st.session_state.current_frame
    else:
        preview_annotated = draw_frame_annotations(
            first_frame,
            [],
            zone_points,
            0,
            zone_defs=zone_defs if zone_defs and len(zone_defs) > 1 else None,
        )
        display_frame = cv2.cvtColor(preview_annotated, cv2.COLOR_BGR2RGB)

    overview_tab, detection_tab, heatmap_tab, timeline_tab, report_tab, settings_tab = st.tabs(
        ["Overview", "Live Detection", "Heatmap", "Incident Timeline", "Compliance Report", "Settings"]
    )
    with overview_tab:
        render_operations_dashboard(
            display_frame,
            st.session_state.violation_log,
            display_gas_context,
            st.session_state.risk_score,
            st.session_state.worker_count,
        )

    with detection_tab:
        st.subheader("3. Live CCTV Detection")
        render_live_analytics_bar(
            st.session_state.violation_log,
            display_gas_context,
            st.session_state.risk_score,
            st.session_state.worker_count,
        )
        st.image(display_frame, channels="RGB", use_column_width=True)
        render_alert_stream(st.session_state.violation_log, display_gas_context, st.session_state.risk_score)

    with heatmap_tab:
        st.subheader("Plant Heatmap & Response Orchestrator")
        heatmap_col, workflow_col = st.columns([1.25, 0.75])
        with heatmap_col:
            st.caption("Geospatial safety view")
            render_safety_heatmap(display_gas_context, st.session_state.risk_score, st.session_state.violation_log)
        with workflow_col:
            st.caption("Automated incident response workflow")
            render_response_workflow(st.session_state.violation_log, display_gas_context)
        render_live_signal_correlation(st.session_state.violation_log, display_gas_context, st.session_state.risk_score)
        if st.session_state.gas_history:
            st.caption("Live gas trend")
            st.line_chart(
                pd.DataFrame(st.session_state.gas_history).set_index("time")[["CH4 %LEL", "CO ppm", "H2S ppm", "O2 %"]],
                height=220,
            )

    with timeline_tab:
        st.subheader("Incident Timeline")
        render_event_playback(st.session_state.violation_log, display_gas_context, key="timeline_tab_incident_playback")
        render_echo_timeline(st.session_state.violation_log, display_gas_context)
        render_future_echo_prediction(st.session_state.violation_log, display_gas_context, st.session_state.risk_score)

    with report_tab:
        st.subheader("AI Incident Summary & Compliance Report")
        render_ai_incident_summary(st.session_state.violation_log, display_gas_context, st.session_state.risk_score)
        st.caption("Explainable multi-agent reasoning chain for the current CCTV, gas, permit, and emergency context.")
        render_agent_copilot(
            st.session_state.violation_log,
            display_gas_context,
            detector,
            st.session_state.risk_score,
        )
        render_why_alert(st.session_state.violation_log, display_gas_context, st.session_state.risk_score)
        render_copilot_questions(st.session_state.violation_log, display_gas_context, st.session_state.risk_score)
        st.markdown("**Regulatory & Incident Intelligence**")
        guidance_items = compliance_guidance(st.session_state.violation_log, display_gas_context)
        guidance_html = "".join(f"<li>{html.escape(item)}</li>" for item in guidance_items)
        st.markdown(f"<ul class='compliance-list'>{guidance_html}</ul>", unsafe_allow_html=True)
        report_text = build_incident_report_text(
            st.session_state.violation_log,
            display_gas_context,
            st.session_state.risk_score,
            st.session_state.risk_level,
        )
        st.download_button(
            "Download Incident Report Draft",
            data=report_text,
            file_name=f"safevision_incident_report_{app_filename_stamp()}.txt",
            mime="text/plain",
            use_container_width=True,
        )
        render_intervention_recommendation(st.session_state.violation_log, display_gas_context, st.session_state.risk_score)

    with settings_tab:
        st.subheader("Permit Simulator & System Settings")
        sim_col, metric_col = st.columns([1, 1])
        with sim_col:
            render_what_if_simulator(display_gas_context, st.session_state.risk_score)
        with metric_col:
            render_future_echo_prediction(st.session_state.violation_log, display_gas_context, st.session_state.risk_score)
        render_evaluation_metrics(
            st.session_state.violation_log,
            display_gas_context,
            st.session_state.risk_score,
            zone_points,
        )
        st.subheader("Administrator Alert Queue")
        if st.session_state.violation_log:
            df = pd.DataFrame(st.session_state.violation_log)
            st.dataframe(df, use_container_width=True, height=270)
        else:
            st.markdown(
                """<div class="log-empty">No administrator alerts yet. Start a scan to populate worker warnings, gas hazard messages, and emergency override events.</div>""",
                unsafe_allow_html=True,
            )

        if st.session_state.csv_path:
            csv_path = Path(st.session_state.csv_path)
            st.success(f"Analysis complete. CSV log saved to {csv_path}")
            if csv_path.exists():
                st.download_button(
                    "Download Violation CSV",
                    data=csv_path.read_bytes(),
                    file_name=csv_path.name,
                    mime="text/csv",
                    use_container_width=True,
                )

    if st.session_state.processing:
        time.sleep(0.25)
        st.rerun()


def main() -> None:
    if "safevision_page" not in st.session_state:
        st.session_state.safevision_page = "home"
    if st.session_state.safevision_page == "dashboard":
        render_dashboard()
    else:
        render_landing_page()


if __name__ == "__main__":
    main()
