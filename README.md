# Multi-Object Detection & Persistent ID Tracking

> **Assignment:** AI / Computer Vision / Data Science  
> **Pipeline:** YOLOv8m → ByteTrack → Annotated MP4

---

## Demo  video link
https://drive.google.com/file/d/1XX8Bss009Yz-3xhuxi6PTxbf-Nh-luId/view?usp=drive_link

| Annotated Output | Movement Heatmap |
|:---:|:---:|
| *(screenshot from output/tracked.mp4)* | *(output/heatmap.jpg)* |

**Source video:** `https://www.youtube.com/shorts/KRoSohvCyok`
** output video:** 'https://drive.google.com/file/d/1XTxGb3wras6fKNSwz7XvrAEI9pcybgW2/view?usp=sharing'

---

## Quick Start

### 1. Clone / unzip the project

```bash
cd sports_tracker
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users (NVIDIA):** also run:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```
> Then set `DEVICE = "cuda"` in `config.py`.

### 4. Download a video

```bash
pip install yt-dlp
yt-dlp -f "bestvideo[height<=720]+bestaudio" -o "input/video.%(ext)s" YOUR_YOUTUBE_URL
```

Update `VIDEO_SOURCE_URL` in `config.py` with the URL you used.

### 5. Test detection first

```bash
python test_detect.py
```

Open `screenshots/detection_test.jpg` to verify YOLOv8 is finding people correctly.

### 6. Run the full pipeline

```bash
# Full video
python main.py

# Quick 300-frame test
python main.py --test

# Custom paths
python main.py --video input/myvideo.mp4 --output output/result.mp4
```

---

## Project Structure

```
sports_tracker/
├── main.py           # Entry point — runs full pipeline
├── detector.py       # YOLOv8 detection wrapper
├── tracker.py        # ByteTrack / DeepSORT tracker
├── visualizer.py     # Bounding box + ID + trail drawing
├── utils.py          # Video I/O, heatmap, plots, progress bar
├── config.py         # All parameters (edit this to tune)
├── test_detect.py    # Quick detection sanity check
├── requirements.txt
├── README.md
├── input/            # Put your video here
├── output/           # Annotated video, heatmap, count plot
└── screenshots/      # Auto-saved frames + detection test
```

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `MODEL_SIZE` | `yolov8m.pt` | Model variant: n/s/m/l/x |
| `CONFIDENCE` | `0.30` | Detection threshold (lower = more detections) |
| `IOU_THRESHOLD` | `0.45` | Non-max suppression IOU |
| `CLASSES` | `[0]` | COCO class IDs (0 = person) |
| `TRACK_BUFFER` | `30` | Frames to keep lost track alive |
| `FRAME_SKIP` | `1` | Process every N frames |
| `SHOW_TRAILS` | `True` | Draw movement trajectories |
| `DEVICE` | `"cpu"` | `"cpu"` \| `"cuda"` \| `"mps"` |

---

## Model & Tracker Choices

### Detector — YOLOv8m

- Pre-trained on COCO-2017 (80 classes, class 0 = person)
- Medium variant chosen for speed/accuracy balance (~45 FPS on GPU, ~8 FPS on CPU)
- Auto-downloads weights on first run (no manual setup)
- Alternative tested: `yolov8n` (faster, less accurate for small/distant players)

### Tracker — ByteTrack

- Implemented via the `supervision` library
- Uses Kalman filter for motion prediction between frames
- Two-pass association: high-confidence detections first, then low-confidence
- This second pass is why ByteTrack handles occlusion better than SORT
- `track_buffer=30`: keeps a lost tracklet alive for 1 second (at 30fps) before dropping the ID

### Why this combination?

ByteTrack requires no appearance model (pure motion), making it fast and robust to camera motion. For sports footage with rapid player movement and frequent partial occlusions, this outperforms simpler SORT trackers. If the video had players with identical jerseys (same appearance), DeepSORT with Re-ID would be preferred.

---

## Outputs

| File | Description |
|---|---|
| `output/tracked.mp4` | Full annotated video with bounding boxes + IDs |
| `output/heatmap.jpg` | Colour map of player positions across the video |
| `output/count_over_time.png` | Plot of active track count per frame |
| `screenshots/` | Auto-captured frames at 10%, 40%, 70%, 95% |

---

## Assumptions

1. Video contains human subjects (uses COCO person class)
2. Camera is mostly static or has slow pan/tilt (fast zoom breaks IoU-based tracking)
3. Players are not smaller than ~20×20 pixels in the frame
4. Video is in standard MP4/MKV/AVI format

---

## Limitations

- **Long occlusion (>1s):** If a player is fully hidden for more than `track_buffer` frames, a new ID is assigned when they reappear
- **Identical appearance:** Players with the same jersey colour may be confused when they cross paths (ByteTrack is motion-only, no Re-ID)
- **Motion blur:** Fast sprints cause missed detections, which can break tracks briefly
- **Camera zoom:** Rapid zoom changes the apparent IoU between frames, causing association failures
- **CPU speed:** On CPU-only machines, expect 5–10 FPS processing (video is still produced correctly, just slowly)

---

## Possible Improvements

1. Add Re-ID embeddings (switch to DeepSORT or BoT-SORT) for players with similar appearance
2. Fine-tune YOLOv8 on domain-specific sports data (e.g. Roboflow cricket datasets)
3. Implement team clustering by jersey colour histogram
4. Add bird's-eye view projection using homography for top-down movement maps
5. Speed estimation using known court dimensions as scale reference

---

## Dependencies

- `ultralytics` — YOLOv8
- `supervision` — ByteTracker + detection utilities
- `opencv-python` — video reading, drawing, writing
- `numpy` — array operations
- `matplotlib` — count-over-time plot
- `yt-dlp` — video download
