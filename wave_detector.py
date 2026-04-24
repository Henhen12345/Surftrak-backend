import cv2
import numpy as np
from typing import List, Tuple
from ultralytics import YOLO

WAVE_VELOCITY_THRESHOLD = 15.0
MIN_WAVE_DURATION_SECONDS = 3.0
WAVE_PAD_SECONDS = 2.5
WAVE_GAP_SECONDS = 4.0
SAMPLE_FPS = 2


def _make_kalman() -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf


def detect_waves(video_path: str) -> List[Tuple[float, float]]:
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    frame_step = max(1, int(fps / SAMPLE_FPS))
    kf = _make_kalman()
    initialized = False

    # Collect (timestamp, velocity_magnitude) pairs
    samples: List[Tuple[float, float]] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            timestamp = frame_idx / fps
            results = model(frame, classes=[0], verbose=False)
            best_box = None
            best_area = 0.0
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    area = (x2 - x1) * (y2 - y1)
                    if area > best_area:
                        best_area = area
                        best_box = (x1, y1, x2, y2)

            if best_box is not None:
                cx = (best_box[0] + best_box[2]) / 2.0
                cy = (best_box[1] + best_box[3]) / 2.0
                if not initialized:
                    kf.statePost = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
                    initialized = True
                kf.correct(np.array([[cx], [cy]], dtype=np.float32))

            predicted = kf.predict()
            vx = float(predicted[2])
            vy = float(predicted[3])
            vel_mag = np.sqrt(vx ** 2 + vy ** 2)
            samples.append((timestamp, vel_mag))

        frame_idx += 1

    cap.release()

    if not samples:
        return []

    # Find contiguous segments where velocity > threshold
    active_segments: List[Tuple[float, float]] = []
    seg_start: float | None = None

    for i, (t, vel) in enumerate(samples):
        if vel >= WAVE_VELOCITY_THRESHOLD:
            if seg_start is None:
                seg_start = t
        else:
            if seg_start is not None:
                seg_end = samples[i - 1][0]
                active_segments.append((seg_start, seg_end))
                seg_start = None

    if seg_start is not None:
        active_segments.append((seg_start, samples[-1][0]))

    # Filter by minimum duration
    valid = [(s, e) for s, e in active_segments if (e - s) >= MIN_WAVE_DURATION_SECONDS]

    if not valid:
        return []

    # Pad segments
    padded = [
        (max(0.0, s - WAVE_PAD_SECONDS), min(duration, e + WAVE_PAD_SECONDS))
        for s, e in valid
    ]

    # Merge segments that are close together
    merged: List[Tuple[float, float]] = [padded[0]]
    for s, e in padded[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e <= WAVE_GAP_SECONDS:
            merged[-1] = (prev_s, max(prev_e, e))
        else:
            merged.append((s, e))

    return merged
