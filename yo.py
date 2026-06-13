import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# =========================
# PAGE CONFIG (ONLY ONCE)
# =========================
st.set_page_config(
    page_title="AI Vehicle Detection System",
    page_icon="🚗",
    layout="wide"
)

# =========================
# LOAD MODEL (CACHE)
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =========================
# TITLE
# =========================
st.title("🚗 AI Vehicle Detection System (Stable Version)")
st.write("Image + Webcam Detection using YOLOv8")

# =========================
# SIDEBAR
# =========================
option = st.sidebar.selectbox(
    "Select Mode",
    ["Image Detection", "Webcam Detection"]
)

# =========================
# IMAGE DETECTION
# =========================
if option == "Image Detection":

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        img_array = np.array(image)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        results = model(img_array)
        output = results[0].plot()

        st.image(output, caption="Detected Objects", use_container_width=True)

# =========================
# WEBCAM DETECTION (LOCAL ONLY)
# =========================
elif option == "Webcam Detection":

    st.warning("⚠ Webcam Streamlit Cloud par work nahi karega. Sirf Local PC par chalega.")

    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:

        ret, frame = camera.read()

        if not ret:
            st.error("Camera not working")
            break

        results = model(frame)
        frame = results[0].plot()

        FRAME_WINDOW.image(frame, channels="BGR")

    camera.release()

