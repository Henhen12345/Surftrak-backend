import math
from typing import Optional, Tuple

import cv2
import numpy as np

from config import (
    CROP_CENTER_ALPHA,
    FRAME_HEIGHT_PX,
    FRAME_WIDTH_PX,
    KALMAN_MEASUREMENT_NOISE,
    KALMAN_PROCESS_NOISE,
    LEAD_CLAMP,
    LEAD_FACTOR,
    MAX_CROP_HEIGHT_FRACTION,
    MIN_CROP_HEIGHT_FRACTION,
    OUTPUT_ASPECT_H,
    OUTPUT_ASPECT_W,
    TARGET_SUBJECT_HEIGHT_RATIO,
    ZOOM_ALPHA,
)
from pipeline.types import BoundingBox, CropRect


def _lerp(current: float, target: float, alpha: float) -> float:
    return current + alpha * (target - current)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class CropEngine:
    """
    Consumes per-frame BoundingBox detections and produces a smooth CropRect
    for every frame of the output video.

    When bbox is None (skipped detection frame or fallback), the Kalman filter
    coasts on its velocity estimate — no correction, position extrapolated forward.
    """

    def __init__(
        self,
        frame_width: int = FRAME_WIDTH_PX,
        frame_height: int = FRAME_HEIGHT_PX,
    ) -> None:
        self._fw = frame_width
        self._fh = frame_height

        # Smoothed state for exponential filtering
        self._smoothed_cx = float(frame_width) / 2.0
        self._smoothed_cy = float(frame_height) / 2.0
        self._smoothed_h = float(frame_height) * 0.60   # conservative initial guess
        self._initialized = False

        self._kf = self._build_kalman()

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, bbox: Optional[BoundingBox]) -> CropRect:
        """
        Advance the engine by one frame.  Feed a detection if available;
        otherwise the Kalman filter coasts on velocity.
        Returns the crop rect to use for this frame.
        """
        if bbox is not None:
            measurement = np.array([[bbox.cx], [bbox.cy]], dtype=np.float32)
            self._kf.predict()
            self._kf.correct(measurement)
        else:
            self._kf.predict()

        state = self._kf.statePost
        kx = float(state[0])
        ky = float(state[1])
        vx = float(state[2])
        vy = float(state[3])

        # Approximate crop size for lead-room calculation (use smoothed value)
        crop_h_approx = self._smoothed_h
        crop_w_approx = crop_h_approx * OUTPUT_ASPECT_W / OUTPUT_ASPECT_H

        # Velocity-based lead room: offset crop center in direction of travel
        lead_x = _clamp(vx * LEAD_FACTOR, -crop_w_approx * LEAD_CLAMP, crop_w_approx * LEAD_CLAMP)
        lead_y = _clamp(vy * LEAD_FACTOR, -crop_h_approx * LEAD_CLAMP, crop_h_approx * LEAD_CLAMP)
        target_cx = kx + lead_x
        target_cy = ky + lead_y

        # Derive target crop height from current detection; hold last if coasting
        if bbox is not None:
            target_crop_h = bbox.h / TARGET_SUBJECT_HEIGHT_RATIO
        else:
            target_crop_h = self._smoothed_h

        target_crop_h = _clamp(
            target_crop_h,
            self._fh * MIN_CROP_HEIGHT_FRACTION,
            self._fh * MAX_CROP_HEIGHT_FRACTION,
        )

        # Snap on first valid frame; smooth thereafter
        if not self._initialized:
            self._smoothed_cx = target_cx
            self._smoothed_cy = target_cy
            self._smoothed_h = target_crop_h
            self._initialized = True
        else:
            self._smoothed_cx = _lerp(self._smoothed_cx, target_cx, CROP_CENTER_ALPHA)
            self._smoothed_cy = _lerp(self._smoothed_cy, target_cy, CROP_CENTER_ALPHA)
            self._smoothed_h = _lerp(self._smoothed_h, target_crop_h, ZOOM_ALPHA)

        return self._build_crop_rect()

    def get_velocity(self) -> Tuple[float, float]:
        """Return the Kalman-estimated (vx, vy) in pixels/frame."""
        state = self._kf.statePost
        return float(state[2]), float(state[3])

    # ── private ───────────────────────────────────────────────────────────────

    def _build_crop_rect(self) -> CropRect:
        crop_h = int(self._smoothed_h)
        crop_w = int(crop_h * OUTPUT_ASPECT_W / OUTPUT_ASPECT_H)

        # Hard clamp to frame bounds
        crop_w = min(crop_w, self._fw)
        crop_h = min(crop_h, self._fh)

        x1 = int(self._smoothed_cx - crop_w / 2)
        y1 = int(self._smoothed_cy - crop_h / 2)
        x1 = int(_clamp(x1, 0, self._fw - crop_w))
        y1 = int(_clamp(y1, 0, self._fh - crop_h))

        return CropRect(x1=x1, y1=y1, w=crop_w, h=crop_h)

    def _build_kalman(self) -> cv2.KalmanFilter:
        """4-state Kalman filter: position (cx, cy) + velocity (vx, vy)."""
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32
        )
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32
        )
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KALMAN_MEASUREMENT_NOISE
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        # Seed state at frame center with zero velocity
        kf.statePost = np.array(
            [[self._fw / 2.0], [self._fh / 2.0], [0.0], [0.0]], dtype=np.float32
        )
        return kf
