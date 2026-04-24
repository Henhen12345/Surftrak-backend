import cv2
import numpy as np
from ultralytics import YOLO

LEAD_FACTOR = 0.15
LEAD_CLAMP = 0.10
CROP_CENTER_ALPHA = 0.10
ZOOM_ALPHA = 0.03
TARGET_SUBJECT_HEIGHT_RATIO = 0.40
DETECT_EVERY_N_FRAMES = 3
OUTPUT_ASPECT = (9, 16)


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


def _compute_crop(
    cx: float,
    cy: float,
    subject_h: float,
    frame_w: int,
    frame_h: int,
    vx: float,
    vy: float,
) -> tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) crop box clamped to frame bounds."""
    crop_h = int(subject_h / TARGET_SUBJECT_HEIGHT_RATIO)
    crop_w = int(crop_h * OUTPUT_ASPECT[0] / OUTPUT_ASPECT[1])

    # Velocity-based lead room
    speed = max(abs(vx), 1e-6)
    lead_x = np.clip(vx / speed * LEAD_FACTOR * crop_w, -LEAD_CLAMP * crop_w, LEAD_CLAMP * crop_w)
    lead_y = np.clip(vy / max(abs(vy), 1e-6) * LEAD_FACTOR * crop_h, -LEAD_CLAMP * crop_h, LEAD_CLAMP * crop_h) if abs(vy) > 1e-6 else 0.0

    center_x = cx + lead_x
    center_y = cy + lead_y

    x1 = int(center_x - crop_w / 2)
    y1 = int(center_y - crop_h / 2)
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # Clamp to frame
    x1 = max(0, min(x1, frame_w - crop_w))
    y1 = max(0, min(y1, frame_h - crop_h))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    return x1, y1, x2, y2


def run_pipeline(input_path: str, output_path: str, job_store: dict, job_id: str) -> None:
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    crop_h_out = frame_h
    crop_w_out = int(crop_h_out * OUTPUT_ASPECT[0] / OUTPUT_ASPECT[1])
    if crop_w_out > frame_w:
        crop_w_out = frame_w
        crop_h_out = int(crop_w_out * OUTPUT_ASPECT[1] / OUTPUT_ASPECT[0])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_w_out, crop_h_out))

    kf = _make_kalman()
    tracker = cv2.TrackerCSRT_create()
    tracker_initialized = False

    # Smoothed state
    smooth_cx = frame_w / 2.0
    smooth_cy = frame_h / 2.0
    smooth_subject_h = frame_h * TARGET_SUBJECT_HEIGHT_RATIO

    last_vx = 0.0
    last_vy = 0.0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected_cx: float | None = None
        detected_cy: float | None = None
        detected_h: float | None = None

        if frame_idx % DETECT_EVERY_N_FRAMES == 0:
            results = model(frame, classes=[0], verbose=False)
            best_box = None
            best_area = 0.0
            for r in results:
                for box in r.boxes:
                    x1b, y1b, x2b, y2b = box.xyxy[0].tolist()
                    area = (x2b - x1b) * (y2b - y1b)
                    if area > best_area:
                        best_area = area
                        best_box = (x1b, y1b, x2b, y2b)

            if best_box is not None:
                x1b, y1b, x2b, y2b = best_box
                detected_cx = (x1b + x2b) / 2.0
                detected_cy = (y1b + y2b) / 2.0
                detected_h = y2b - y1b

                # Re-init CSRT tracker
                tracker = cv2.TrackerCSRT_create()
                bbox = (int(x1b), int(y1b), int(x2b - x1b), int(y2b - y1b))
                tracker.init(frame, bbox)
                tracker_initialized = True

                # Feed Kalman measurement
                if kf.statePre is None or np.all(kf.statePre == 0):
                    kf.statePost = np.array([[detected_cx], [detected_cy], [0.0], [0.0]], dtype=np.float32)
                kf.correct(np.array([[detected_cx], [detected_cy]], dtype=np.float32))
        else:
            # Use CSRT tracker between detections
            if tracker_initialized:
                ok, bbox = tracker.update(frame)
                if ok:
                    tx, ty, tw, th = bbox
                    detected_cx = tx + tw / 2.0
                    detected_cy = ty + th / 2.0
                    detected_h = float(th)
                    kf.correct(np.array([[detected_cx], [detected_cy]], dtype=np.float32))

        # Kalman predict always
        predicted = kf.predict()
        pred_cx = float(predicted[0])
        pred_cy = float(predicted[1])
        pred_vx = float(predicted[2])
        pred_vy = float(predicted[3])

        if detected_cx is not None:
            last_vx = pred_vx
            last_vy = pred_vy
            use_cx = detected_cx
            use_cy = detected_cy
            use_h = detected_h
        else:
            use_cx = pred_cx
            use_cy = pred_cy
            use_h = smooth_subject_h

        # Exponential smoothing
        smooth_cx = CROP_CENTER_ALPHA * use_cx + (1 - CROP_CENTER_ALPHA) * smooth_cx
        smooth_cy = CROP_CENTER_ALPHA * use_cy + (1 - CROP_CENTER_ALPHA) * smooth_cy
        smooth_subject_h = ZOOM_ALPHA * use_h + (1 - ZOOM_ALPHA) * smooth_subject_h

        x1, y1, x2, y2 = _compute_crop(
            smooth_cx, smooth_cy, smooth_subject_h,
            frame_w, frame_h, last_vx, last_vy,
        )

        cropped = frame[y1:y2, x1:x2]
        if cropped.shape[0] == 0 or cropped.shape[1] == 0:
            cropped = frame[0:crop_h_out, 0:crop_w_out]

        resized = cv2.resize(cropped, (crop_w_out, crop_h_out))
        out.write(resized)

        frame_idx += 1
        if total_frames > 0:
            job_store[job_id]["progress"] = min(frame_idx / total_frames, 0.99)

    cap.release()
    out.release()
