import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from config import (
    FRAME_HEIGHT_PX,
    FRAME_WIDTH_PX,
    LOCK_BREAK_FRAMES,
    LOCK_CONFIRMATION_FRAMES,
    MAX_SUBJECT_HEIGHT_FRACTION,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    MIN_LATERAL_VELOCITY_PX_PER_FRAME,
    MIN_SUBJECT_HEIGHT_FRACTION,
    MOG2_DETECT_SHADOWS,
    MOG2_HISTORY,
    MOG2_VAR_THRESHOLD,
    UWB_SEARCH_STRIP_WIDTH_FRACTION,
    VELOCITY_HISTORY_FRAMES,
    WATER_ZONE_TOP_FRACTION,
)
from pipeline.types import BoundingBox

logger = logging.getLogger(__name__)

# Minimum motion contour area; below this is sensor noise or sea spray, not a person
_MIN_CONTOUR_AREA = 600
# 7×7 kernel connects body parts (arms, board) into one contour blob
_MORPH_KERNEL_SIZE = 7
# Padding around each motion candidate before running pose estimation
_CANDIDATE_PAD_PX = 15
# Landmark indices: shoulders (11,12), hips (23,24), ankles (27,28)
_KEY_LANDMARK_IDS = [11, 12, 23, 24, 27, 28]
# Minimum visibility to include a landmark in the bounding box computation
_LANDMARK_VISIBILITY_THRESHOLD = 0.3
# Two detections within this many horizontal pixels are treated as the same candidate
_CANDIDATE_MATCH_RADIUS_PX = 80
# Prune stale candidate history every N frames to cap memory on long sessions
_HISTORY_PRUNE_INTERVAL = 300
# Minimum composite score to begin the confirmation countdown for a new lock
_MIN_CONFIRM_SCORE = 0.35
# Velocity normalisation ceiling: 8 px/frame maps to a velocity score of 1.0
_VELOCITY_SCORE_CEILING = 8.0


@dataclass
class _Candidate:
    """Internal working state for one pose-confirmed detection in a frame."""
    bbox: BoundingBox
    candidate_id: int
    lateral_velocity: float = 0.0
    total_score: float = 0.0


class SubjectFinder:
    """
    Locates the active surfer in each frame using a cascade of filters designed
    to reject beach spectators, photographers, and bystanders.

    Cascade (each step narrows the candidate pool):
      MOG2 motion mask
        → water zone filter   (rejects beach / sky detections)
        → size filter         (rejects spectators who are too close or too far)
        → UWB strip filter    (rejects surfers who are not the tagged subject)
        → MediaPipe Pose      (rejects non-human blobs, produces tight bbox)
        → velocity filter     (rejects stationary or slow-moving people)
        → lock state machine  (requires sustained motion before committing)

    The three-state machine (searching → confirming → locked) prevents snapping
    to a briefly-moving spectator. A subject must sustain LOCK_CONFIRMATION_FRAMES
    of qualifying detections before the lock is committed.
    """

    def __init__(self) -> None:
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=MOG2_DETECT_SHADOWS,
        )
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_MORPH_KERNEL_SIZE, _MORPH_KERNEL_SIZE)
        )

        # ── Three-state lock machine ───────────────────────────────────────────
        self._lock_state: str = "searching"        # "searching" | "confirming" | "locked"
        self._locked_bbox: Optional[BoundingBox] = None
        self._locked_candidate_id: Optional[int] = None
        self._frames_since_last_detection: int = 0
        self._confirmation_frame_count: int = 0    # counts toward LOCK_CONFIRMATION_FRAMES
        self._break_frame_count: int = 0           # counts toward LOCK_BREAK_FRAMES

        # ── Per-candidate velocity history ─────────────────────────────────────
        # Key: stable candidate ID. Value: deque of (frame_idx, center_x) tuples.
        self._candidate_history: Dict[int, deque] = {}
        self._next_candidate_id: int = 0

        self._pose = self._init_mediapipe()

    # ── public ────────────────────────────────────────────────────────────────

    def find_subject(
        self,
        frame: np.ndarray,
        uwb_predicted_col: int,
        frame_idx: int,
    ) -> Optional[BoundingBox]:
        """
        Process one frame and return the surfer's BoundingBox, or None.

        None means no confident detection this frame. The CropEngine's Kalman
        filter will coast forward — preferable to an incorrect lock.
        """
        fh, fw = frame.shape[:2]

        # Prune stale candidate IDs periodically to cap memory on long sessions
        if frame_idx % _HISTORY_PRUNE_INTERVAL == 0 and frame_idx > 0:
            self._prune_candidate_history(frame_idx)

        # ── Step 1: UWB search strip ───────────────────────────────────────────
        # UWB narrows the horizontal search region. Without it we search the full
        # width, raising false-positive risk in busy lineups.
        half_strip = int(FRAME_WIDTH_PX * UWB_SEARCH_STRIP_WIDTH_FRACTION / 2)
        search_left = max(0, uwb_predicted_col - half_strip)
        search_right = min(FRAME_WIDTH_PX, uwb_predicted_col + half_strip)
        uwb_active = (search_right - search_left) < FRAME_WIDTH_PX

        # ── Step 2: Motion mask (full frame) ──────────────────────────────────
        # MOG2 must see the full frame to build an accurate background model.
        # We filter the CONTOURS we consider, not the frame fed to MOG2.
        # 7×7 morphological close joins body parts into a single contour blob.
        fg_mask = self._mog2.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self._morph_kernel)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_rects: List[tuple] = [
            cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > _MIN_CONTOUR_AREA
        ]

        # ── Step 3: Water zone filter ──────────────────────────────────────────
        # WATER ZONE FILTER: The beach, rocks, and sky occupy the upper portion of
        # the frame. Any detection whose centre is above this line is a spectator
        # or photographer — never an in-water surfer.
        # Tune WATER_ZONE_TOP_FRACTION in config.py to match your spot's horizon.
        water_top_px = int(FRAME_HEIGHT_PX * WATER_ZONE_TOP_FRACTION)
        motion_rects = [
            r for r in motion_rects if (r[1] + r[3] / 2) >= water_top_px
        ]

        # ── Step 4: Size filter ────────────────────────────────────────────────
        # SIZE FILTER: Spectators close to camera appear very tall (> 35% frame
        # height). Distant noise / sea spray appears very small (< 4% frame height).
        # At typical SurfTrak distances (15–80 ft) the surfer occupies 4–35%.
        min_h = int(FRAME_HEIGHT_PX * MIN_SUBJECT_HEIGHT_FRACTION)
        max_h = int(FRAME_HEIGHT_PX * MAX_SUBJECT_HEIGHT_FRACTION)
        motion_rects = [
            r for r in motion_rects if min_h <= r[3] <= max_h
        ]

        # ── Step 5: UWB strip filter ───────────────────────────────────────────
        # UWB STRIP FILTER: Discard candidates whose horizontal centre falls outside
        # the UWB-predicted column ± search strip. Rejects other surfers in the
        # background who are not the UWB-tagged subject.
        if uwb_active:
            motion_rects = [
                r for r in motion_rects if search_left <= (r[0] + r[2] / 2) <= search_right
            ]

        # ── Step 6: MediaPipe Pose on surviving candidates ─────────────────────
        # By this point typically 0–3 blobs remain. Running MediaPipe only on
        # survivors (not the full frame) is significantly faster and yields a
        # precise landmark-derived bounding box for each human shape.
        raw_candidates: List[_Candidate] = []
        for (mx, my, mw, mh) in motion_rects:
            bbox = self._run_mediapipe_on_region(frame, mx, my, mw, mh, fw, fh)
            if bbox is None:
                continue
            cid = self._match_or_create_candidate(bbox.cx, frame_idx)
            raw_candidates.append(_Candidate(bbox=bbox, candidate_id=cid))

        # ── Step 7: Lateral velocity per candidate ─────────────────────────────
        for cand in raw_candidates:
            history = self._candidate_history[cand.candidate_id]
            if len(history) >= 3:
                frames_seen = [h[0] for h in history]
                xs = [h[1] for h in history]
                elapsed = max(1, frames_seen[-1] - frames_seen[0])
                cand.lateral_velocity = abs(xs[-1] - xs[0]) / elapsed
            # len < 3: velocity stays 0.0 → will not pass the filter on first sighting

        # ── Step 8: Velocity filter ────────────────────────────────────────────
        # VELOCITY FILTER: Active surfers move laterally fast — they're on a wave.
        # Spectators, photographers, and idle paddlers in the lineup fail this test.
        # EXCEPTION: if the candidate is already our locked subject, do NOT discard
        # on velocity alone — use the LOCK_BREAK_FRAMES grace period instead.
        # This preserves the lock through stalls, wipeouts, and kick-outs.
        survivors: List[_Candidate] = [
            c for c in raw_candidates
            if c.lateral_velocity >= MIN_LATERAL_VELOCITY_PX_PER_FRAME
            or c.candidate_id == self._locked_candidate_id
        ]

        # ── Step 9: Scoring + lock state machine ───────────────────────────────
        return self._apply_state_machine(survivors, uwb_predicted_col)

    # ── private: state machine ─────────────────────────────────────────────────

    def _apply_state_machine(
        self,
        candidates: List[_Candidate],
        uwb_predicted_col: int,
    ) -> Optional[BoundingBox]:
        """
        Three-state machine:  searching → confirming → locked

        Returns a BoundingBox only in "locked" state (or while coasting on last
        known position). During "searching" and "confirming" returns None so the
        CropEngine coasts — this avoids framing the wrong person for the first
        LOCK_CONFIRMATION_FRAMES of a wave.
        """
        if not candidates:
            return self._handle_no_candidates()

        # Score each survivor: UWB proximity + normalised lateral speed
        for cand in candidates:
            proximity = max(0.0, 1.0 - abs(cand.bbox.cx - uwb_predicted_col) / (FRAME_WIDTH_PX / 2))
            vel_score = min(1.0, cand.lateral_velocity / _VELOCITY_SCORE_CEILING)
            cand.total_score = 0.6 * proximity + 0.4 * vel_score

        candidates.sort(key=lambda c: c.total_score, reverse=True)
        best = candidates[0]

        if self._lock_state == "searching":
            if best.total_score >= _MIN_CONFIRM_SCORE:
                # Start confirmation countdown on this candidate
                self._lock_state = "confirming"
                self._confirmation_frame_count = 1
                self._locked_candidate_id = best.candidate_id
                self._locked_bbox = best.bbox
            return None

        elif self._lock_state == "confirming":
            if best.candidate_id == self._locked_candidate_id:
                self._confirmation_frame_count += 1
                self._locked_bbox = best.bbox
                if self._confirmation_frame_count >= LOCK_CONFIRMATION_FRAMES:
                    self._lock_state = "locked"
                    self._break_frame_count = 0
                    logger.info(
                        "SubjectFinder: lock confirmed on candidate %d after %d frames.",
                        self._locked_candidate_id,
                        self._confirmation_frame_count,
                    )
            else:
                # A different candidate scored best — restart confirmation
                self._lock_state = "searching"
                self._confirmation_frame_count = 0
            return None   # never emit output during confirmation

        else:  # "locked"
            if best.candidate_id == self._locked_candidate_id:
                # Solid lock — update bbox and reset break counter
                self._locked_bbox = best.bbox
                self._frames_since_last_detection = 0
                self._break_frame_count = 0
                return self._locked_bbox
            else:
                # Best visible candidate is not our locked subject
                return self._handle_locked_no_detection()

    def _handle_no_candidates(self) -> Optional[BoundingBox]:
        """Called when no candidates survived the filter cascade."""
        if self._lock_state == "locked":
            return self._handle_locked_no_detection()
        # In searching or confirming — nothing to act on
        self._lock_state = "searching"
        return None

    def _handle_locked_no_detection(self) -> Optional[BoundingBox]:
        """
        Called when we are locked but the locked candidate was not found this frame.
        Coasts on the last known bbox until LOCK_BREAK_FRAMES is exhausted,
        then releases the lock so we can re-acquire.
        """
        self._break_frame_count += 1
        self._frames_since_last_detection += 1

        if self._break_frame_count >= LOCK_BREAK_FRAMES:
            logger.info(
                "SubjectFinder: lock released after %d consecutive missed frames.",
                self._break_frame_count,
            )
            self._lock_state = "searching"
            self._locked_candidate_id = None
            self._locked_bbox = None
            return None

        return self._locked_bbox   # coast — CropEngine Kalman will extrapolate

    # ── private: detection helpers ─────────────────────────────────────────────

    def _run_mediapipe_on_region(
        self,
        frame: np.ndarray,
        mx: int, my: int, mw: int, mh: int,
        fw: int, fh: int,
    ) -> Optional[BoundingBox]:
        """Run MediaPipe Pose on a padded crop. Returns full-frame bbox or None."""
        x1 = max(0, mx - _CANDIDATE_PAD_PX)
        y1 = max(0, my - _CANDIDATE_PAD_PX)
        x2 = min(fw, mx + mw + _CANDIDATE_PAD_PX)
        y2 = min(fh, my + mh + _CANDIDATE_PAD_PX)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        results = self._pose.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None

        lms = results.pose_landmarks.landmark
        roi_h, roi_w = roi.shape[:2]

        # Use mean visibility of shoulder/hip/ankle landmarks as pose confidence
        key_vis = [lms[i].visibility for i in _KEY_LANDMARK_IDS]
        if float(np.mean(key_vis)) < MEDIAPIPE_MIN_DETECTION_CONFIDENCE:
            return None

        # Build full-frame bounding box from all sufficiently visible landmarks
        visible_pts = [
            (lm.x * roi_w + x1, lm.y * roi_h + y1)
            for lm in lms
            if lm.visibility > _LANDMARK_VISIBILITY_THRESHOLD
        ]
        if len(visible_pts) < 2:
            return None

        xs = [p[0] for p in visible_pts]
        ys = [p[1] for p in visible_pts]
        return BoundingBox(
            x=int(min(xs)),
            y=int(min(ys)),
            w=max(1, int(max(xs) - min(xs))),
            h=max(1, int(max(ys) - min(ys))),
        )

    def _match_or_create_candidate(self, center_x: float, frame_idx: int) -> int:
        """
        Assign a stable ID to a detection by matching it to the nearest existing
        candidate within _CANDIDATE_MATCH_RADIUS_PX. Creates a new ID if no match.
        """
        for cid, history in self._candidate_history.items():
            if history and abs(center_x - history[-1][1]) < _CANDIDATE_MATCH_RADIUS_PX:
                history.append((frame_idx, center_x))
                return cid

        new_id = self._next_candidate_id
        self._next_candidate_id += 1
        self._candidate_history[new_id] = deque(
            [(frame_idx, center_x)], maxlen=VELOCITY_HISTORY_FRAMES
        )
        return new_id

    def _prune_candidate_history(self, frame_idx: int) -> None:
        """Remove candidate entries not seen within the last VELOCITY_HISTORY_FRAMES."""
        stale = [
            cid for cid, hist in self._candidate_history.items()
            if not hist or (frame_idx - hist[-1][0]) > VELOCITY_HISTORY_FRAMES
        ]
        for cid in stale:
            del self._candidate_history[cid]
        if stale:
            logger.debug("SubjectFinder: pruned %d stale candidate(s).", len(stale))

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
