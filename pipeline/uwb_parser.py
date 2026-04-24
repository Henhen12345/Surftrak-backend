import json
import math
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from config import (
    CAMERA_HFOV_DEGREES,
    FRAME_WIDTH_PX,
)

logger = logging.getLogger(__name__)

# Interpolation is skipped when two readings are this far apart in time.
# Wild extrapolation across sensor gaps causes sudden crop jumps.
_MAX_INTERPOLATION_GAP_S = 0.5


class UWBParser:
    """
    Parses a UWB ranging log and converts per-timestamp distances into
    predicted horizontal pixel columns for the subject's position.

    Fails silently on any parse error — callers check is_available() and
    fall back to full-frame CV search if the log is unavailable.
    """

    def __init__(self, uwb_log_path: Optional[str]) -> None:
        # List of (timestamp_seconds, predicted_column_px) tuples, sorted by time
        self._readings: List[Tuple[float, int]] = []
        self._available = False

        if uwb_log_path is None:
            logger.info("No UWB log provided — falling back to full-frame CV search.")
            return

        try:
            self._parse(Path(uwb_log_path))
            if self._readings:
                self._available = True
                logger.info("UWB log loaded: %d valid readings.", len(self._readings))
            else:
                logger.warning("UWB log contained no valid readings.")
        except Exception as exc:
            logger.warning("UWB log parse failed (%s). Falling back to full-frame CV.", exc)

    # ── public ────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        return self._available

    def get_predicted_column(self, video_timestamp_seconds: float) -> int:
        """
        Return the predicted horizontal pixel column of the subject at the
        given video timestamp.  Linearly interpolates between the two nearest
        readings.  Returns the frame center if UWB data is unavailable.
        """
        if not self._available:
            return FRAME_WIDTH_PX // 2

        # Before the first reading → assume center (camera hasn't locked on yet)
        if video_timestamp_seconds <= self._readings[0][0]:
            return FRAME_WIDTH_PX // 2

        # After the last reading → hold last known column
        if video_timestamp_seconds >= self._readings[-1][0]:
            return self._readings[-1][1]

        # Binary search for the two bracketing readings
        lo, hi = 0, len(self._readings) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self._readings[mid][0] <= video_timestamp_seconds:
                lo = mid
            else:
                hi = mid

        t0, col0 = self._readings[lo]
        t1, col1 = self._readings[hi]

        # Don't interpolate across a sensor gap — return last known column
        if t1 - t0 > _MAX_INTERPOLATION_GAP_S:
            return col0

        alpha = (video_timestamp_seconds - t0) / (t1 - t0)
        interpolated = col0 + alpha * (col1 - col0)
        return int(_clamp(interpolated, 0, FRAME_WIDTH_PX - 1))

    # ── private ───────────────────────────────────────────────────────────────

    def _parse(self, path: Path) -> None:
        with path.open("r") as fh:
            data = json.load(fh)

        baseline_cm = float(data["anchor_baseline_cm"])
        if baseline_cm <= 0:
            raise ValueError("anchor_baseline_cm must be positive")

        for reading in data["readings"]:
            t = float(reading["t"])
            d1 = float(reading["d1_cm"])
            d2 = float(reading["d2_cm"])

            # Skip invalid or sentinel values
            if any(math.isnan(v) or math.isinf(v) for v in (t, d1, d2)):
                continue
            if d1 <= 0 or d2 <= 0:
                continue

            col = self._distances_to_column(d1, d2, baseline_cm)
            self._readings.append((t, col))

        # Ensure chronological order (firmware should already guarantee this)
        self._readings.sort(key=lambda r: r[0])

    @staticmethod
    def _distances_to_column(d1_cm: float, d2_cm: float, baseline_cm: float) -> int:
        """
        Convert anchor distances to a predicted horizontal pixel column.
        d1 = left anchor distance, d2 = right anchor distance.
        Positive angle → surfer is to the right of center.
        """
        delta = d1_cm - d2_cm
        # arcsin(delta/baseline) gives the horizontal pan angle
        angle_rad = math.asin(_clamp(delta / baseline_cm, -1.0, 1.0))
        angle_deg = math.degrees(angle_rad)

        # Map angle to [0, FRAME_WIDTH_PX]: 0° = center, ±(HFOV/2) = edges
        normalized = angle_deg / (CAMERA_HFOV_DEGREES / 2)   # -1.0 … 1.0
        col = int((normalized + 1.0) / 2.0 * FRAME_WIDTH_PX)
        return int(_clamp(col, 0, FRAME_WIDTH_PX - 1))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
