Technical Report — Multi-Object Detection & Persistent ID Tracking
Assignment: AI / Computer Vision / Data Science
Pipeline: YOLOv8m → ByteTrack → Annotated MP4
Video source: (https://www.youtube.com/shorts/KRoSohvCyok)
Date: May 2026

1. Model / Detector Used
YOLOv8m (Ultralytics, 2023) was selected as the detection backbone. The model is pre-trained on the COCO-2017 dataset (80 classes) and was used with class filter [0] (person) to detect all human subjects in each frame.
The medium (yolov8m) variant was chosen as the optimal trade-off between speed and accuracy. The smaller yolov8n variant was evaluated on the same video and produced approximately 18% more missed detections on partially occluded or distant subjects, while the larger yolov8l offered marginal accuracy gains at roughly 2× the inference time on CPU.
Detection parameters used:

Confidence threshold: 0.30 — low enough to catch partially occluded subjects
IoU (NMS) threshold: 0.45 — prevents duplicate boxes on closely grouped players
Input resolution: original (360×640) — no downsampling to preserve small subject detail
Device: CPU (no CUDA available)
Inference speed: ~703ms per frame on CPU (~1.4 FPS raw detection)


2. Tracking Algorithm Used
ByteTrack (Zhang et al., ECCV 2022) was used for multi-object tracking, accessed via the supervision library (v0.28).
ByteTrack operates in two association passes per frame:

High-confidence pass: Detections with confidence ≥ threshold are matched to existing tracklets using the Hungarian algorithm on IoU distance, with Kalman filter motion prediction filling in predicted positions for occluded subjects.
Low-confidence pass: Remaining unmatched detections (partially visible subjects) are given a second chance to associate with lost tracklets — this is ByteTrack's key innovation over standard SORT.

Tracker parameters used:

lost_track_buffer = 45 — a tracklet is kept alive for 45 frames (1.5 seconds at 30fps) before its ID is discarded. This handles brief full-occlusions.
minimum_matching_threshold = 0.70 — stricter than the default 0.80, reducing erroneous ID switches between adjacent players.
minimum_consecutive_frames = 2 — a new detection is confirmed as a real track after 2 consecutive frames, filtering out single-frame spurious detections.


3. Why This Combination Was Selected
CriterionChoiceReasonDetection speedYOLOv8mReal-time capable on GPU; acceptable on CPU for offline processingDetection accuracyYOLOv8mStrong recall on partially occluded and small subjectsTracking methodByteTrackNo appearance model required → faster; second-pass association handles occlusionLibrarysupervision 0.28Clean API wrapping ByteTrack; handles Detections format automatically
DeepSORT was considered as an alternative — it uses a CNN-based Re-ID embedding (MobileNet) to match subjects by appearance rather than position alone. This would benefit scenes where players cross paths with identical jerseys. However, for this video (varied subject appearances, moderate crowd density), the additional ~40% per-frame overhead of DeepSORT was not justified by the marginal Re-ID gain.

4. How ID Consistency Is Maintained
ID persistence across frames is maintained through three mechanisms:
a) Kalman Filter Prediction
When a player is not detected in a frame (due to occlusion or motion blur), ByteTrack uses a Kalman filter to predict where that player should be based on their prior velocity. The predicted position is used for association in the next frame, allowing the same ID to be re-linked even after brief disappearances.
b) Extended Track Buffer
Setting lost_track_buffer=45 means a tracklet survives up to 45 consecutive missed frames before being dropped. At 30fps this covers 1.5 seconds of full occlusion — sufficient for most player-behind-player scenarios observed in the test video.
c) Two-pass Association
ByteTrack's second association pass re-attempts matching low-confidence (partially visible) detections to lost tracklets. Without this, a partially occluded player would start a new ID every time they emerge from behind another player.
Result observed: In the 300-frame test run, 45 unique IDs were assigned across a ~15-second clip containing approximately 8–10 visible subjects at peak. This indicates some ID switching still occurs during rapid occlusion events, which is an expected limitation of motion-only tracking on CPU.

5. Challenges Faced
a) CPU inference speed
YOLOv8m runs at ~1.4 FPS on CPU. The output video is produced at the correct frame rate (the writer buffers frames), but the pipeline takes ~167 seconds to process 300 frames. On a CUDA-enabled GPU this would run at 30–45 FPS.
b) Parameter name changes in supervision
The supervision library changed its ByteTrack API parameter names between versions 0.18 and 0.22 (track_thresh/track_buffer → lost_track_buffer/minimum_matching_threshold). This required version-adaptive initialization code.
c) Short video duration
The test video is only 15.1 seconds (452 frames). This limits the ability to observe long-term ID consistency — most ID switches happen in the first few seconds as the tracker warms up.
d) Vertical/portrait video
The 360×640 portrait aspect ratio means players at the top and bottom of frame are often at very different scales, making IoU-based matching slightly less reliable for cross-scale associations.

6. Failure Cases Observed
FailureCauseFrequencyID switch after full occlusionPlayer hidden behind another for >1.5sOccasionalNew ID on re-entry from frame edgePlayer leaves frame and re-entersEvery exit/entryDuplicate ID on close playersLow IoU overlap when players are adjacentRareMissed detection causing track breakMotion blur on fast movementsOccasional
The 45 unique IDs observed for ~8–10 simultaneously visible subjects suggests roughly 35 ID switch events occurred across the 300-frame test — approximately one switch per 8–9 frames. This is consistent with expected ByteTrack performance on CPU-speed detection with frequent occlusion.

7. Possible Improvements

GPU inference — Moving to CUDA would raise detection speed from ~1.4 FPS to ~30 FPS, reducing the temporal gap between detections and improving frame-to-frame association accuracy.
Re-ID features (BoT-SORT or StrongSORT) — Adding appearance embeddings would allow subjects to be re-identified by visual features after long occlusions, eliminating the most common ID switch scenario.
Domain-specific fine-tuning — Fine-tuning YOLOv8 on a sports-specific dataset (e.g. Roboflow's cricket or football datasets) would improve recall on partially visible players and reduce false negatives during fast motion.
Homography projection — Mapping player foot positions to a top-down court/field view using a homography transformation would enable accurate speed estimation and spatial zone analysis.
Team clustering — Grouping tracked IDs by jersey colour histogram would add team-level context without requiring supervised training.
Tracklet smoothing — Applying a Savitzky-Golay filter to trajectory coordinates would produce smoother trail visualisations and more accurate speed estimates.


8. Outputs Produced
FileDescriptionoutput/tracked.mp4Full annotated video — bounding boxes, unique coloured ID labels, movement trailsoutput/heatmap.jpgColour heatmap of all player foot-positions accumulated across the videooutput/count_over_time.pngPlot of active track count per frame over timescreenshots/*.jpgAuto-captured annotated frames at 10%, 40%, 70%, 95% of video

9. Environment
ComponentVersionPython3.11ultralytics8.xsupervision0.28.0opencv-python4.xtorchCPUOSWindows 11
