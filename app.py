import streamlit as st
import cv2
from main import process_video  # your function

st.title("Multi-Object Tracking Demo")

uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_file is not None:
    with open("temp.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.write("Processing...")

    output_path = process_video("temp.mp4")

    st.video(output_path)