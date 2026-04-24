import logging
from pathlib import Path
from typing import Callable, List

import cv2
import numpy as np

from pipeline.types import CropRect

logger = logging.getLogger(__name__)

# Standard output resolution for 9:16 vertical video
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920

# Codecs to try in order; avc1 = H.264 on most platforms, mp4v = universal fallback
_CODEC_CANDIDATES = ["avc1", "X264", "H264", "mp4v"]

# Report progress every this many frames to avoid excessive job-store writes
_PROGRESS_REPORT_INTERVAL = 30


class Renderer:
    """
    Writes the framed output video by applying per-frame CropRects to the
    original input and resizing each crop to the standard 1080×1920 output.
    """

    def render(
        self,
        input_path: str,
        output_path: str,
        crop_rects: List[CropRect],
        progress_callback: Callable[[float], None],
    ) -> bool:
        """
        Read every frame from `input_path`, apply the corresponding CropRect,
        and write the resized frame to `output_path`.

        progress_callback receives a float in [0.0, 1.0].
        Returns True on success, False if the output file could not be verified.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error("Renderer: cannot open input video %s", input_path)
            return False

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = self._open_writer(output_path, fps)
        if writer is None:
            cap.release()
            logger.error("Renderer: could not open VideoWriter for %s", output_path)
            return False

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rect = crop_rects[frame_idx] if frame_idx < len(crop_rects) else self._center_crop(frame)

                cropped = self._apply_crop(frame, rect)
                resized = cv2.resize(cropped, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
                writer.write(resized)

                frame_idx += 1
                if frame_idx % _PROGRESS_REPORT_INTERVAL == 0 and total_frames > 0:
                    progress_callback(frame_idx / total_frames)

        except Exception as exc:
            logger.exception("Renderer: error at frame %d — %s", frame_idx, exc)
            return False
        finally:
            cap.release()
            writer.release()

        progress_callback(1.0)

        # Verify the output file was actually written
        out = Path(output_path)
        if not out.exists() or out.stat().st_size == 0:
            logger.error("Renderer: output file missing or empty: %s", output_path)
            return False

        logger.info("Renderer: wrote %d frames → %s (%.1f MB)", frame_idx, output_path, out.stat().st_size / 1e6)
        return True

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_crop(frame: np.ndarray, rect: CropRect) -> np.ndarray:
        fh, fw = frame.shape[:2]
        x1 = max(0, rect.x1)
        y1 = max(0, rect.y1)
        x2 = min(fw, rect.x1 + rect.w)
        y2 = min(fh, rect.y1 + rect.h)
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return frame   # safety fallback: return full frame
        return cropped

    @staticmethod
    def _center_crop(frame: np.ndarray) -> np.ndarray:
        """Fallback center crop when crop_rects list is shorter than frame count."""
        fh, fw = frame.shape[:2]
        crop_w = int(fh * OUTPUT_WIDTH / OUTPUT_HEIGHT)
        x1 = max(0, (fw - crop_w) // 2)
        return frame[:, x1: x1 + crop_w]

    @staticmethod
    def _open_writer(output_path: str, fps: float):
        for codec in _CODEC_CANDIDATES:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
            if writer.isOpened():
                logger.info("Renderer: using codec '%s'.", codec)
                return writer
        return None
