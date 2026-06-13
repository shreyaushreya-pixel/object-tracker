from datetime import datetime
import pandas as pd

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
from datetime import datetime
import av
import cv2
import numpy as np
import pandas as pd
import time

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Vehicle Monitoring System",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 AI Vehicle Tracking & Speed Monitoring System")
st.write("Real-Time Detection + Tracking + Speed + Date/Time Logging")

# =====================================================
# SESSION LOGS
# =====================================================

if "logs" not in st.session_state:
    st.session_state.logs = []

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("⚙ Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    0.10,
    1.00,
    0.50
)

speed_limit = st.sidebar.slider(
    "Speed Limit (km/h)",
    10,
    150,
    60
)

detect_person = st.sidebar.checkbox(
    "Detect Person",
    value=True
)

detect_vehicle = st.sidebar.checkbox(
    "Detect Vehicles",
    value=True
)

# =====================================================
# VIDEO PROCESSOR
# =====================================================

class VehicleProcessor(VideoProcessorBase):

    def __init__(self):
        self.next_id = 0
        self.tracked = {}

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        results = model(img, verbose=False)

        current_time = time.time()

        current_datetime = datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        )

        # Show Date & Time on Screen
        cv2.putText(
            img,
            current_datetime,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        for box in results[0].boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < confidence:
                continue

            label = model.names[cls]

            allowed = []

            if detect_person:
                allowed.append("person")

            if detect_vehicle:
                allowed.extend([
                    "car",
                    "truck",
                    "bus",
                    "motorcycle"
                ])

            if label not in allowed:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            matched_id = None

            # =====================================
            # TRACKING
            # =====================================

            for vid, data in self.tracked.items():

                px, py = data["center"]

                dist = np.sqrt(
                    (center_x - px) ** 2 +
                    (center_y - py) ** 2
                )

                if dist < 60:
                    matched_id = vid
                    break

            if matched_id is None:
                self.next_id += 1
                matched_id = self.next_id

            speed = 0

            if matched_id in self.tracked:

                prev_x, prev_y = self.tracked[matched_id]["center"]
                prev_t = self.tracked[matched_id]["time"]

                dt = current_time - prev_t

                if dt > 0:

                    pixel_distance = np.sqrt(
                        (center_x - prev_x) ** 2 +
                        (center_y - prev_y) ** 2
                    )

                    speed = pixel_distance / dt * 0.15

            self.tracked[matched_id] = {
                "center": (center_x, center_y),
                "time": current_time
            }

            # =====================================
            # COLOR
            # =====================================

            color = (0, 255, 0)

            if speed > speed_limit:
                color = (0, 0, 255)

            # =====================================
            # DRAW BOX
            # =====================================

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.circle(
                img,
                (center_x, center_y),
                5,
                (255, 0, 0),
                -1
            )

            text = (
                f"{label} "
                f"{conf:.2f} "
                f"ID:{matched_id} "
                f"{int(speed)}km/h"
            )

            cv2.putText(
                img,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            # =====================================
            # LOGGING
            # =====================================

            st.session_state.logs.append({
                "DateTime": current_datetime,
                "Object": label,
                "ID": matched_id,
                "Speed": int(speed)
            })

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

# =====================================================
# CAMERA
# =====================================================

st.subheader("📷 Live Camera")

webrtc_streamer(
    key="vehicle-monitor",
    video_processor_factory=VehicleProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)

# =====================================================
# REPORT TABLE
# =====================================================

st.subheader("📄 Detection Logs")

if st.session_state.logs:

    df = pd.DataFrame(st.session_state.logs)

    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False)

    st.download_button(
        "⬇ Download CSV Report",
        csv,
        "vehicle_detection_report.csv",
        "text/csv"
    )

# =====================================================
# FOOTER
# =====================================================

st.success("✅ System Ready")       


