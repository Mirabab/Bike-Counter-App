
import streamlit as st
import pandas as pd
import cv2
import tempfile
from ultralytics import YOLO
import altair as alt
import os

st.title("🚲 AI-Powered Bike Counter")

video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    st.video(video_path)
    st.write("🔍 Detecting bikes... Please wait.")

    model = YOLO("yolov8n.pt")  # You can switch to a custom model for better results
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        count = 0

        for result in results:
            # DEBUG: show raw results
            st.write("Detections:", result.boxes.cls.tolist())

            # Fallback: count all detected objects temporarily
            count += len(result.boxes)

        timestamp = frame_count / fps
        data.append({"timestamp_sec": timestamp, "bike_count": count})
        frame_count += 1

    cap.release()

    if not data:
        st.error("No data was captured during detection.")
    else:
        df = pd.DataFrame(data)
        df["minute"] = df["timestamp_sec"] // 60

        st.success("✅ Analysis Complete")
        st.subheader("📈 Bike Count Over Time")
        st.line_chart(df.set_index("timestamp_sec")["bike_count"])

        st.download_button("Download JSON", df.to_json(orient="records"), "bike_counts.json")
        st.download_button("Download CSV", df.to_csv(index=False), "bike_counts.csv")

        st.altair_chart(
            alt.Chart(df).mark_line().encode(
                x='timestamp_sec',
                y='bike_count'
            ).properties(title="Bike Count Over Time"),
            use_container_width=True
        )
