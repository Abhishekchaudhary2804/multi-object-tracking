import streamlit as st
import os
import config
from main import run_pipeline

st.title("Multi-Object Tracking Demo")

uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_file is not None:
    temp_input = "temp_input.mp4"

    # Save uploaded file
    with open(temp_input, "wb") as f:
        f.write(uploaded_file.read())

    st.write("Processing video...")

    try:
        result = run_pipeline(
            video_path=temp_input,
            output_path=config.OUTPUT_PATH,
            max_frames=None,
            show_trails=True
        )

        st.success(
            f"✓ Processing complete! Processed {result['frames']} frames, detected {result['unique_ids']} unique objects."
        )

        # Use the absolute output path so Streamlit can locate it reliably.
        output_path = os.path.abspath(config.OUTPUT_PATH)
        if os.path.exists(output_path):
            st.video(output_path)
        else:
            st.error("Output video not found!")

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")

    finally:
        if os.path.exists(temp_input):
            os.remove(temp_input)