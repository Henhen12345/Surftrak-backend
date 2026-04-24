import logging
import math
from typing import List

import numpy as np
from scipy.signal import medfilt

from config import (
    MIN_WAVE_DURATION_SECONDS,
    WAVE_GAP_MERGE_SECONDS,
    WAVE_PAD_SECONDS,
    WAVE_VELOCITY_THRESHOLD,
)
from models import WaveMetadata
from pipeline.types import CropRect

logger = logging.getLogger(__name__)

# Median filter window for velocity smoothing (must be odd)
_VELOCITY_SMOOTH_WINDOW = 15
# Thumbnail is extracted from 30% into the wave — captures the pop-up moment
_THUMBNAIL_POSITION = 0.30


class WaveDetector:
    """
    Detects individual wave rides from the crop-rect velocity time series.

    Uses the frame-to-frame displacement of crop centers as a proxy for surfer
    speed.  A sustained period of high velocity (> WAVE_VELOCITY_THRESHOLD)
    that lasts long enough is classified as a wave ride.

    Runs entirely in-memory from the crop_rects list produced during the
    detection pass — no second video decode required.
    """

    def detect_waves(
        self,
        crop_rects: List[CropRect],
        fps: float,
    ) -> List[WaveMetadata]:
        if len(crop_rects) < 2:
            logger.info("WaveDetector: too few frames to analyse.")
            return []

        total_duration = len(crop_rects) / fps

        # ── Step 1: Velocity time series from crop center displacement ──────────
        velocities = np.zeros(len(crop_rects), dtype=np.float32)
        for i in range(1, len(crop_rects)):
            dx = crop_rects[i].center_x - crop_rects[i - 1].center_x
            dy = crop_rects[i].center_y - crop_rects[i - 1].center_y
            velocities[i] = math.sqrt(dx * dx + dy * dy)

        # Rolling median suppresses single-frame Kalman spikes
        window = min(_VELOCITY_SMOOTH_WINDOW, len(velocities))
        if window % 2 == 0:
            window -= 1
        window = max(3, window)
        smoothed = medfilt(velocities, kernel_size=window)

        # ── Step 2: Binary signal → contiguous high-velocity segments ──────────
        is_riding = smoothed > WAVE_VELOCITY_THRESHOLD
        raw_segments: List[tuple] = []
        in_seg = False
        seg_start = 0

        for i, riding in enumerate(is_riding):
            if riding and not in_seg:
                seg_start = i
                in_seg = True
            elif not riding and in_seg:
                raw_segments.append((seg_start / fps, i / fps))
                in_seg = False

        if in_seg:
            raw_segments.append((seg_start / fps, len(crop_rects) / fps))

        # ── Step 3: Filter by minimum qualifying duration ───────────────────────
        valid = [
            (s, e)
            for s, e in raw_segments
            if (e - s) >= MIN_WAVE_DURATION_SECONDS
        ]

        if not valid:
            logger.info("WaveDetector: no qualifying wave segments found.")
            return []

        # ── Step 4: Merge segments that are separated by a small gap ───────────
        merged: List[tuple] = [valid[0]]
        for s, e in valid[1:]:
            prev_s, prev_e = merged[-1]
            if s - prev_e <= WAVE_GAP_MERGE_SECONDS:
                merged[-1] = (prev_s, max(prev_e, e))
            else:
                merged.append((s, e))

        # ── Step 5: Pad and build final WaveMetadata list ───────────────────────
        durations = [e - s for s, e in merged]
        best_idx = int(np.argmax(durations))

        result: List[WaveMetadata] = []
        for i, (s, e) in enumerate(merged):
            start = round(max(0.0, s - WAVE_PAD_SECONDS), 3)
            end = round(min(total_duration, e + WAVE_PAD_SECONDS), 3)
            duration = round(end - start, 3)
            result.append(
                WaveMetadata(
                    index=i,
                    start_seconds=start,
                    end_seconds=end,
                    duration_seconds=duration,
                    thumbnail_offset=round(start + duration * _THUMBNAIL_POSITION, 3),
                    is_best_wave=(i == best_idx),
                )
            )

        result.sort(key=lambda w: w.start_seconds)
        logger.info("WaveDetector: detected %d wave(s).", len(result))
        return result
