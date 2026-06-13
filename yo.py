import streamlit as st

st.title("Test App")

try:
    import cv2
    st.success("OpenCV Working")
except Exception as e:
    st.error(f"OpenCV Error: {e}")

try:
    from ultralytics import YOLO
    st.success("YOLO Working")
except Exception as e:
    st.error(f"YOLO Error: {e}")

