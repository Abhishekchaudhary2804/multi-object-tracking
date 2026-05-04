import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from collections import defaultdict, deque

st.set_page_config(page_title="Sports Object Tracker", page_icon="🎯", layout="wide")

st.title("🎯 Multi-Object Detection & Persistent ID Tracking")
st.markdown("Upload a sports/event video — detects all people with **YOLOv8** and tracks them with **ByteTrack**.")

# ── Sidebar params ────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    conf         = st.slider("Confidence threshold", 0.1, 0.9, 0.30, 0.05)
    iou          = st.slider("IoU (NMS) threshold",  0.1, 0.9, 0.45, 0.05)
    track_buffer = st.slider("Track buffer (frames)", 10, 90, 45)
    match_thresh = st.slider("Match threshold",       0.3, 0.95, 0.70, 0.05)
    show_trails  = st.toggle("Show movement trails", True)
    max_frames   = st.number_input("Max frames (0 = all)", 0, 10000, 300)

# ── Color helper ──────────────────────────────────────────────
PALETTE = [
    (255,80,80),(80,200,255),(80,255,120),(255,180,40),
    (200,80,255),(255,120,200),(40,220,200),(255,220,60),
]
def get_color(tid): return PALETTE[int(tid) % len(PALETTE)]

# ── Model loader (cached) ─────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLOv8…")
def load_model():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")   # nano = fastest on CPU

# ── Draw function ─────────────────────────────────────────────
def draw_tracks(frame, tracked, trails=None):
    if trails:
        for tid, pts in trails.items():
            if len(pts) < 2: continue
            color = get_color(tid)
            n = len(pts)
            for i in range(1, n):
                a = i / n
                cv2.line(frame, pts[i-1], pts[i],
                         tuple(int(c*a) for c in color), max(1, int(2*a)))

    if tracked is None or len(tracked) == 0:
        return frame

    try:
        tids = tracked.tracker_id
        if tids is None: return frame
        for xyxy, tid in zip(tracked.xyxy, tids):
            x1,y1,x2,y2 = map(int, xyxy)
            color = get_color(tid)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            label = f"ID {tid}"
            (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
            cv2.putText(frame, label, (x1+3, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    except AttributeError:
        pass
    return frame

# ── Main pipeline ─────────────────────────────────────────────
def process_video(video_path, model, conf, iou, track_buffer,
                  match_thresh, show_trails, max_frames, prog, status):
    import supervision as sv

    # Init tracker — handle both old and new supervision API
    try:
        tracker = sv.ByteTrack(
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            minimum_consecutive_frames=2,
        )
    except TypeError:
        try:
            tracker = sv.ByteTrack(track_buffer=track_buffer, match_thresh=match_thresh)
        except TypeError:
            tracker = sv.ByteTrack()

    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(total, max_frames) if max_frames > 0 else total

    tmp_out = tempfile.mktemp(suffix=".mp4")
    writer  = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    trails        = defaultdict(lambda: [])
    count_history = []
    all_ids       = set()
    heatmap       = np.zeros((H, W), dtype=np.float32)
    screenshots   = []
    shot_at       = {int(total*0.25), int(total*0.5), int(total*0.75)}
    frame_n       = 0

    while cap.isOpened() and frame_n < total:
        ret, frame = cap.read()
        if not ret: break

        # Detect
        res   = model.predict(frame, conf=conf, iou=iou, classes=[0], verbose=False)[0]
        boxes = res.boxes.data.cpu().numpy() if res.boxes and len(res.boxes) else np.empty((0,6))

        # Track
        if len(boxes) > 0:
            det     = sv.Detections(
                xyxy=boxes[:,:4].astype(np.float32),
                confidence=boxes[:,4].astype(np.float32),
                class_id=boxes[:,5].astype(int),
            )
            tracked = tracker.update_with_detections(det)
        else:
            tracked = sv.Detections.empty()

        # Update trails & heatmap
        active = 0
        if tracked is not None and len(tracked) > 0 and tracked.tracker_id is not None:
            active = len(tracked.tracker_id)
            for xyxy, tid in zip(tracked.xyxy, tracked.tracker_id):
                cx = int((xyxy[0]+xyxy[2])/2)
                cy = int(xyxy[3])
                if show_trails:
                    trails[int(tid)].append((cx, cy))
                    if len(trails[int(tid)]) > 40:
                        trails[int(tid)] = trails[int(tid)][-40:]
                all_ids.add(int(tid))
                heatmap[min(cy,H-1), min(cx,W-1)] += 1

        count_history.append(active)

        frame = draw_tracks(frame, tracked, trails if show_trails else None)
        if frame_n in shot_at:
            screenshots.append(frame.copy())

        writer.write(frame)
        frame_n += 1
        prog.progress(frame_n / total)
        status.text(f"Frame {frame_n}/{total} — {active} active tracks")

    cap.release()
    writer.release()

    # Heatmap image
    blurred  = cv2.GaussianBlur(heatmap, (51,51), 0)
    norm     = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    hm_color = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)

    return dict(video=tmp_out, counts=count_history, ids=len(all_ids),
                fps=fps, frames=frame_n, heatmap=hm_color, shots=screenshots)

# ── UI ────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload video (MP4, AVI, MOV)", type=["mp4","avi","mov","mkv"])

if uploaded:
    tmp_in = tempfile.mktemp(suffix=".mp4")
    with open(tmp_in, "wb") as f:
        f.write(uploaded.read())

    cap = cv2.VideoCapture(tmp_in)
    fps_v = cap.get(cv2.CAP_PROP_FPS)
    W_v   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_v   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tot_v = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Resolution", f"{W_v}×{H_v}")
    c2.metric("FPS", f"{fps_v:.0f}")
    c3.metric("Frames", tot_v)
    c4.metric("Duration", f"{tot_v/max(fps_v,1):.1f}s")

    if st.button("▶ Run Tracking Pipeline", type="primary", use_container_width=True):
        model = load_model()
        prog  = st.progress(0)
        status= st.empty()
        t0    = time.time()

        res = process_video(tmp_in, model, conf, iou, track_buffer,
                            match_thresh, show_trails, max_frames, prog, status)

        elapsed = time.time() - t0
        prog.empty(); status.empty()
        st.success(f"✅ Done in {elapsed:.1f}s — {res['ids']} unique IDs tracked")

        tab1, tab2, tab3, tab4 = st.tabs(["📹 Video", "🔥 Heatmap", "📊 Count Plot", "🖼️ Screenshots"])

        with tab1:
            with open(res["video"], "rb") as vf:
                st.download_button("⬇ Download tracked.mp4", vf,
                                   "tracked.mp4", "video/mp4", use_container_width=True)
            st.info("Download the video and open it locally to play it.")

        with tab2:
            st.image(cv2.cvtColor(res["heatmap"], cv2.COLOR_BGR2RGB),
                     caption="Movement heatmap — red = most visited area",
                     use_container_width=True)

        with tab3:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12,3))
            times = [i/res["fps"] for i in range(len(res["counts"]))]
            ax.fill_between(times, res["counts"], alpha=0.3, color="#40a0ff")
            ax.plot(times, res["counts"], color="#40a0ff", linewidth=1.5)
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Active Tracks")
            ax.set_title("Tracked Subjects Over Time")
            ax.grid(axis="y", alpha=0.3)
            st.pyplot(fig); plt.close()

        with tab4:
            if res["shots"]:
                cols = st.columns(len(res["shots"]))
                for col, shot in zip(cols, res["shots"]):
                    col.image(cv2.cvtColor(shot, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.info("Video too short for screenshots.")

    if os.path.exists(tmp_in):
        os.remove(tmp_in)

else:
    st.info("👆 Upload a video above to get started.")
    st.markdown("""
    **Pipeline:** YOLOv8n (detection) → ByteTrack (tracking) → Annotated MP4
    
    **Outputs:** Annotated video · Movement heatmap · Track count plot · Frame screenshots
    """)