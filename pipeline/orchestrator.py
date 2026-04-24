import logging
import traceback
from pathlib import Path
from typing import Optional

import cv2

from config import FRAME_HEIGHT_PX, FRAME_WIDTH_PX, PROCESS_EVERY_N_FRAMES
from job_store import JobStore
from pipeline.crop_engine import CropEngine
from pipeline.renderer import Renderer
from pipeline.subject_finder import SubjectFinder
from pipeline.uwb_parser import UWBParser
from pipeline.wave_detector import WaveDetector

logger = logging.getLogger(__name__)


def run_pipeline(
    job_id: str,
    input_video_path: str,
    uwb_log_path: Optional[str],
    output_video_path: str,
    job_store: JobStore,
) -> None:
    """
    Top-level pipeline coordinator.  Runs in a ThreadPoolExecutor worker thread.

    Progress is partitioned across stages:
      0.00–0.02  startup / UWB parsing
      0.02–0.08  video validation
      0.08–0.63  subject detection + crop computation (frame loop)
      0.63–0.88  rendering
      0.88–0.98  wave detection
      0.98–1.00  finalisation
    """
    try:
        # ── Stage 1: Parse UWB log ─────────────────────────────────────────────
        job_store.update_progress(job_id, 0.02, "Parsing UWB data")
        uwb = UWBParser(uwb_log_path)
        if not uwb.is_available():
            logger.warning("Job %s: UWB unavailable — using full-frame CV search.", job_id)

        # ── Stage 2: Validate input video ──────────────────────────────────────
        job_store.update_progress(job_id, 0.05, "Validating video")
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open input video: {input_video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total_frames == 0:
            raise RuntimeError("Video reports zero frames — file may be corrupt.")
        duration_seconds = total_frames / fps
        logger.info("Job %s: %.1fs @ %.2f fps (%d frames).", job_id, duration_seconds, fps, total_frames)

        # ── Stage 3: Detection pass — subject bounding boxes + crop rects ──────
        job_store.update_progress(job_id, 0.08, "Detecting subject")
        finder = SubjectFinder()
        engine = CropEngine(FRAME_WIDTH_PX, FRAME_HEIGHT_PX)
        crop_rects = []

        cap = cv2.VideoCapture(input_video_path)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            video_ts = frame_idx / fps

            if frame_idx % PROCESS_EVERY_N_FRAMES == 0:
                # Full detection: UWB narrows the search strip, MediaPipe finds pose
                uwb_col = uwb.get_predicted_column(video_ts)
                bbox = finder.find_subject(frame, uwb_col, frame_idx)
            else:
                # Skipped frame — Kalman coasts forward using its velocity estimate
                bbox = None

            crop_rect = engine.update(bbox)
            crop_rects.append(crop_rect)
            frame_idx += 1

            if frame_idx % 60 == 0 and total_frames > 0:
                stage_progress = 0.08 + (frame_idx / total_frames) * 0.55
                job_store.update_progress(job_id, stage_progress, "Detecting subject")

        cap.release()

        # Guard: crop_rects must have exactly one entry per frame
        if len(crop_rects) != frame_idx:
            raise RuntimeError(
                f"crop_rects length mismatch: {len(crop_rects)} vs {frame_idx} frames"
            )
        logger.info("Job %s: detection pass complete (%d crop rects).", job_id, len(crop_rects))

        # ── Stage 4: Render output video ───────────────────────────────────────
        job_store.update_progress(job_id, 0.63, "Rendering output")

        def render_progress(p: float) -> None:
            job_store.update_progress(job_id, 0.63 + p * 0.25, "Rendering output")

        success = Renderer().render(
            input_path=input_video_path,
            output_path=output_video_path,
            crop_rects=crop_rects,
            progress_callback=render_progress,
        )
        if not success:
            raise RuntimeError("Renderer failed — output file is missing or empty.")

        # ── Stage 5: Wave detection from crop-rect velocity ────────────────────
        job_store.update_progress(job_id, 0.88, "Detecting waves")
        waves = WaveDetector().detect_waves(crop_rects=crop_rects, fps=fps)

        # ── Stage 6: Finalise ──────────────────────────────────────────────────
        job_store.update_progress(job_id, 0.98, "Finalising")
        best_wave_index: Optional[int] = None
        if waves:
            best_wave_index = next(
                (i for i, w in enumerate(waves) if w.is_best_wave), None
            )

        job_store.mark_complete(
            job_id=job_id,
            wave_count=len(waves),
            best_wave_index=best_wave_index,
            duration_seconds=round(duration_seconds, 3),
            waves=[w.dict() for w in waves],
        )
        logger.info(
            "Job %s complete — %d wave(s), best=%s.", job_id, len(waves), best_wave_index
        )

    except MemoryError:
        # Long videos can exhaust RAM on Railway free tier — give a useful message
        msg = (
            "MemoryError: video may be too large for the current server tier. "
            f"Duration was approximately {duration_seconds:.0f}s."
            if "duration_seconds" in dir()
            else "MemoryError: video may be too large."
        )
        logger.error("Job %s: %s", job_id, msg)
        job_store.mark_failed(job_id, msg)

    except Exception:
        detail = traceback.format_exc()
        logger.error("Job %s failed:\n%s", job_id, detail)
        job_store.mark_failed(job_id, detail)
