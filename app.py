
import streamlit as st
import pandas as pd
import cv2
import tempfile
from ultralytics import YOLO
import altair as alt
import os

st.set_page_config(layout="wide")
st.title("🚲 AI-Powered Bike Counter")

video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    st.video(video_path)
    progress_bar = st.progress(0)
    st.write("🔍 Detecting bikes... This may take a moment.")

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    data = []
    snapshots = []
    detected_summary = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        count = 0
        for result in results:
            cls_list = result.boxes.cls.tolist()
            detected_summary += cls_list
            count += len(result.boxes)

        timestamp = frame_count / fps
        data.append({"timestamp_sec": timestamp, "bike_count": count})

        # Save snapshot for peak detection later
        if count >= 3:
            _, buffer = cv2.imencode(".jpg", frame)
            snapshots.append({"timestamp_sec": timestamp, "image": buffer.tobytes(), "count": count})

        frame_count += 1
        progress_bar.progress(min(frame_count / total_frames, 1.0))

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

        # Compact detected class summary
        st.markdown("### 🔎 Detected Class Summary")
        from collections import Counter
        count_summary = Counter([int(c) for c in detected_summary])
        st.code(str(dict(count_summary)), language="json")

        # Show peak & softest moments
        max_row = df[df.bike_count == df.bike_count.max()].iloc[0]
        min_row = df[df.bike_count == df.bike_count.min()].iloc[0]

        st.markdown("### 🚦 Detection Highlights")
        col1, col2 = st.columns(2)
        col1.metric("📈 Peak Count", f"{int(max_row.bike_count)} bikes", help=f"At {max_row.timestamp_sec:.2f} sec")
        col2.metric("📉 Lowest Count", f"{int(min_row.bike_count)} bikes", help=f"At {min_row.timestamp_sec:.2f} sec")

        # Show images from peak periods
        if snapshots:
            st.markdown("### 🖼️ Snapshot Frames with High Bike Count")
            for snap in sorted(snapshots, key=lambda x: -x["count"])[:3]:  # top 3
                st.image(snap["image"], caption=f"{snap['count']} bikes @ {snap['timestamp_sec']:.2f}s", use_column_width=True)
