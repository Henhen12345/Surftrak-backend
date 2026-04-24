import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import (
    DETECTION_FALLBACK_HOLD_FRAMES,
    FRAME_HEIGHT_PX,
    FRAME_WIDTH_PX,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    MIDLINE_SCORE_WEIGHT,
    MOG2_DETECT_SHADOWS,
    MOG2_HISTORY,
    MOG2_VAR_THRESHOLD,
    UWB_SEARCH_STRIP_WIDTH_FRACTION,
)
from pipeline.types import BoundingBox

logger = logging.getLogger(__name__)

# Minimum contour area to treat as a motion candidate (filters sensor noise)
_MIN_CONTOUR_AREA = 800
# Padding added around each motion candidate before running pose estimation
_CANDIDATE_PAD_PX = 20
# MediaPipe landmark indices used for pose confidence and bounding box
_KEY_LANDMARK_IDS = [11, 12, 23, 24, 27, 28]   # shoulders, hips, ankles
# Minimum visibility threshold to include a landmark in bounding box computation
_LANDMARK_VISIBILITY_THRESHOLD = 0.3
# IoU threshold: if new detection overlaps last bbox this much, it's the same subject
_LOCK_IOU_THRESHOLD = 0.3
# Minimum composite score to establish a new lock on a candidate
_MIN_LOCK_SCORE = 0.4


class SubjectFinder:
    """
    Locates the surfer in each video frame using a two-stage approach:

    1. MOG2 background subtraction narrows the search to moving regions.
    2. MediaPipe Pose precisely locates the subject within each motion blob.

    UWB data restricts the horizontal search strip, dramatically reducing false
    positives from other surfers or whitewash in the wider frame.
    """

    def __init__(self) -> None:
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=MOG2_DETECT_SHADOWS,
        )
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Lock state
        self._has_lock = False
        self._last_known_bbox: Optional[BoundingBox] = None
        self._frames_since_last_detection = 0
        self._locked_subject_id = 0

        self._pose = self._init_mediapipe()

    # ── public ────────────────────────────────────────────────────────────────

    def find_subject(
        self,
        frame: np.ndarray,
        uwb_predicted_col: int,
        frame_idx: int,
    ) -> Optional[BoundingBox]:
        """
        Attempt to locate the surfer in `frame`.

        Returns a BoundingBox in full-frame pixel coordinates, or None if the
        subject cannot be confidently identified (caller should coast on last
        known position via the CropEngine).
        """
        # ── Step 1: Define search strip ────────────────────────────────────────
        half_strip = int(FRAME_WIDTH_PX * UWB_SEARCH_STRIP_WIDTH_FRACTION / 2)
        search_left = max(0, uwb_predicted_col - half_strip)
        search_right = min(FRAME_WIDTH_PX, uwb_predicted_col + half_strip)

        # ── Step 2: Motion mask ────────────────────────────────────────────────
        fg_mask = self._mog2.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self._morph_kernel)

        # Zero out pixels outside the UWB search strip
        strip_mask = np.zeros_like(fg_mask)
        strip_mask[:, search_left:search_right] = fg_mask[:, search_left:search_right]

        contours, _ = cv2.findContours(strip_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_rects = [
            cv2.boundingRect(c)
            for c in contours
            if cv2.contourArea(c) > _MIN_CONTOUR_AREA
        ]

        # ── Step 3: Pose estimation on motion candidates ───────────────────────
        scored: List[Tuple[float, BoundingBox]] = []
        fh, fw = frame.shape[:2]

        for (mx, my, mw, mh) in motion_rects:
            # Expand candidate region with padding for a cleaner pose crop
            x1 = max(0, mx - _CANDIDATE_PAD_PX)
            y1 = max(0, my - _CANDIDATE_PAD_PX)
            x2 = min(fw, mx + mw + _CANDIDATE_PAD_PX)
            y2 = min(fh, my + mh + _CANDIDATE_PAD_PX)

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = self._pose.process(roi_rgb)
            if not results.pose_landmarks:
                continue

            lms = results.pose_landmarks.landmark
            roi_h, roi_w = roi.shape[:2]

            # Use mean visibility of key landmarks as detection confidence proxy
            key_visibilities = [lms[i].visibility for i in _KEY_LANDMARK_IDS]
            confidence = float(np.mean(key_visibilities)) if key_visibilities else 0.0
            if confidence < MEDIAPIPE_MIN_DETECTION_CONFIDENCE:
                continue

            # Build full-frame bounding box from all sufficiently visible landmarks
            visible_pts = [
                (lm.x * roi_w + x1, lm.y * roi_h + y1)
                for lm in lms
                if lm.visibility > _LANDMARK_VISIBILITY_THRESHOLD
            ]
            if len(visible_pts) < 2:
                continue

            pts_x = [p[0] for p in visible_pts]
            pts_y = [p[1] for p in visible_pts]
            bbox = BoundingBox(
                x=int(min(pts_x)),
                y=int(min(pts_y)),
                w=max(1, int(max(pts_x) - min(pts_x))),
                h=max(1, int(max(pts_y) - min(pts_y))),
            )

            # Score: proximity to UWB column + detection confidence
            proximity = 1.0 - abs(bbox.cx - uwb_predicted_col) / (FRAME_WIDTH_PX / 2)
            proximity = max(0.0, proximity)
            score = MIDLINE_SCORE_WEIGHT * proximity + (1.0 - MIDLINE_SCORE_WEIGHT) * confidence
            scored.append((score, bbox))

        # ── Step 4: Subject selection + lock persistence ───────────────────────
        selected: Optional[BoundingBox] = None

        if self._has_lock and self._last_known_bbox is not None:
            # Prefer whichever candidate overlaps the currently tracked subject
            for score, bbox in scored:
                if _iou(self._last_known_bbox, bbox) > _LOCK_IOU_THRESHOLD:
                    selected = bbox
                    break

        if selected is None and scored:
            # No lock or locked subject lost — pick highest-scoring candidate
            scored.sort(key=lambda t: t[0], reverse=True)
            best_score, best_bbox = scored[0]
            if best_score >= _MIN_LOCK_SCORE:
                selected = best_bbox
                if not self._has_lock:
                    self._locked_subject_id += 1
                self._has_lock = True

        if selected is not None:
            self._last_known_bbox = selected
            self._frames_since_last_detection = 0
            return selected

        # ── Step 5: Fallback — coast on last known position ────────────────────
        self._frames_since_last_detection += 1
        if self._frames_since_last_detection < DETECTION_FALLBACK_HOLD_FRAMES:
            return self._last_known_bbox   # CropEngine Kalman will extrapolate

        # Lock expired — signal that we have no usable subject position
        self._has_lock = False
        return None

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _init_mediapipe():
        try:
            import mediapipe as mp
            return mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
            )
        except Exception as exc:
            logger.error("MediaPipe failed to initialise: %s", exc)
            raise


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection-over-union of two bounding boxes."""
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x + a.w, b.x + b.w)
    iy2 = min(a.y + a.h, b.y + b.h)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0
