import asyncio
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

import cv2
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse

from job_store import JobStore
from models import (
    HealthResponse,
    JobStatus,
    ProcessResponse,
    WaveMetadata,
    WavesResponse,
)
from pipeline.orchestrator import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SurfTrak Studio API",
    version="1.0.0",
    description="UWB + CV hybrid framing pipeline for autonomous surf cameras.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Tightened per-origin once app Store domain is known
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── Shared state ─────────────────────────────────────────────────────────────

job_store = JobStore()

# Max 2 concurrent jobs — prevents OOM on Railway free tier (512 MB RAM)
MAX_CONCURRENT_JOBS = 2
_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS)
_active_job_count = 0
_active_job_lock = threading.Lock()

# Thumbnail bytes cached in memory after first extraction (job_id → wave_index → bytes)
_thumbnail_cache: Dict[str, Dict[int, bytes]] = {}

# Tracks whether MediaPipe successfully loaded at startup
_pipeline_ready = False

# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup() -> None:
    global _pipeline_ready
    try:
        import mediapipe as mp
        # Instantiate and immediately close to validate the install
        pose = mp.solutions.pose.Pose(model_complexity=1)
        pose.close()
        _pipeline_ready = True
        logger.info("MediaPipe loaded successfully — pipeline ready.")
    except Exception as exc:
        logger.warning("MediaPipe failed to load on startup: %s", exc)
        _pipeline_ready = False

# ── Helpers ───────────────────────────────────────────────────────────────────

def _decrement_active() -> None:
    global _active_job_count
    with _active_job_lock:
        _active_job_count -= 1


def _job_wrapper(
    job_id: str,
    input_video_path: str,
    uwb_log_path: Optional[str],
    output_video_path: str,
) -> None:
    try:
        run_pipeline(job_id, input_video_path, uwb_log_path, output_video_path, job_store)
    finally:
        _decrement_active()


def _get_job_or_404(job_id: str) -> dict:
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


def _require_complete(job: dict) -> None:
    if job["status"] != "complete":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not complete (status={job['status']}, stage={job['stage']}).",
        )

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/process", response_model=ProcessResponse, status_code=202)
async def process_video(
    video: UploadFile = File(..., description="Raw session video (.mp4 / .mov / .avi / .mkv)"),
    uwb_log: Optional[UploadFile] = File(
        None, description="UWB ranging JSON log from Pi firmware (optional)"
    ),
) -> ProcessResponse:
    """
    Upload a raw surf session video (+ optional UWB log) to start processing.
    Returns a job_id immediately; poll /status/{job_id} for progress.
    """
    global _active_job_count

    # Reject when both worker slots are full rather than queue indefinitely
    with _active_job_lock:
        if _active_job_count >= MAX_CONCURRENT_JOBS:
            raise HTTPException(
                status_code=503,
                detail="Server busy — both processing slots are occupied. Please retry in a moment.",
            )
        _active_job_count += 1

    # Validate file type
    video_name = video.filename or ""
    if not any(video_name.lower().endswith(ext) for ext in (".mp4", ".mov", ".avi", ".mkv")):
        with _active_job_lock:
            _active_job_count -= 1
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format. Accepted: .mp4, .mov, .avi, .mkv",
        )

    job_id = str(uuid.uuid4())
    suffix = Path(video_name).suffix or ".mp4"
    input_path = Path("/tmp") / f"{job_id}_input{suffix}"
    output_path = Path("/tmp") / f"{job_id}_output.mp4"
    uwb_path: Optional[Path] = None

    # Read and persist uploads
    video_bytes = await video.read()
    if len(video_bytes) > 8 * 1024 ** 3:   # 8 GB hard limit
        with _active_job_lock:
            _active_job_count -= 1
        raise HTTPException(status_code=413, detail="Video file exceeds the 8 GB size limit.")

    input_path.write_bytes(video_bytes)

    if uwb_log is not None:
        uwb_path = Path("/tmp") / f"{job_id}_uwb.json"
        uwb_path.write_bytes(await uwb_log.read())

    job_store.create_job(
        job_id=job_id,
        input_video_path=str(input_path),
        uwb_log_path=str(uwb_path) if uwb_path else None,
        output_video_path=str(output_path),
    )

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _executor,
        _job_wrapper,
        job_id,
        str(input_path),
        str(uwb_path) if uwb_path else None,
        str(output_path),
    )

    logger.info("Job %s queued (video=%.1f MB, uwb=%s).", job_id, len(video_bytes) / 1e6, uwb_path is not None)
    return ProcessResponse(
        job_id=job_id,
        status="queued",
        message="Processing started. Poll /status/{job_id} for updates.",
    )


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str) -> JobStatus:
    """Return the current processing status and progress of a job."""
    job = _get_job_or_404(job_id)
    return JobStatus(
        job_id=job["job_id"],
        status=job["status"],
        progress=job["progress"],
        stage=job["stage"],
        wave_count=job["wave_count"],
        best_wave_index=job["best_wave_index"],
        duration_seconds=job["duration_seconds"],
        error_message=job["error_message"],
        created_at=job["created_at"],
        completed_at=job["completed_at"],
    )


@app.get("/download/{job_id}")
async def download_video(job_id: str) -> FileResponse:
    """Stream the framed 9:16 output video. Only available when status = complete."""
    job = _get_job_or_404(job_id)
    _require_complete(job)

    out = Path(job["output_video_path"])
    if not out.exists() or out.stat().st_size == 0:
        raise HTTPException(status_code=500, detail="Output file is missing — the job may need to be re-run.")

    return FileResponse(
        path=str(out),
        media_type="video/mp4",
        filename=f"surftrak_session_{job_id}.mp4",
        headers={"Content-Disposition": f'attachment; filename="surftrak_session_{job_id}.mp4"'},
    )


@app.get("/waves/{job_id}", response_model=WavesResponse)
async def get_waves(job_id: str) -> WavesResponse:
    """Return metadata for each auto-detected wave clip."""
    job = _get_job_or_404(job_id)
    _require_complete(job)

    waves = [WaveMetadata(**w) for w in job["waves"]]
    total_ride_time = round(sum(w.duration_seconds for w in waves), 3)

    return WavesResponse(
        job_id=job_id,
        waves=waves,
        total_session_duration=job["duration_seconds"] or 0.0,
        total_ride_time=total_ride_time,
    )


@app.get("/thumbnail/{job_id}/{wave_index}")
async def get_thumbnail(job_id: str, wave_index: int) -> Response:
    """
    Extract and return a JPEG thumbnail from a specific wave.
    Results are cached in memory after the first request.
    """
    job = _get_job_or_404(job_id)
    _require_complete(job)

    waves = job["waves"]
    if wave_index < 0 or wave_index >= len(waves):
        raise HTTPException(
            status_code=404,
            detail=f"wave_index {wave_index} is out of range (0–{len(waves) - 1}).",
        )

    # Serve from cache if available (critical for iOS wave card loading speed)
    cached = _thumbnail_cache.get(job_id, {}).get(wave_index)
    if cached is not None:
        return Response(content=cached, media_type="image/jpeg")

    thumbnail_offset = waves[wave_index]["thumbnail_offset"]
    out_path = job["output_video_path"]

    jpeg_bytes = _extract_jpeg(out_path, thumbnail_offset)
    if jpeg_bytes is None:
        raise HTTPException(status_code=500, detail="Failed to extract thumbnail frame from output video.")

    _thumbnail_cache.setdefault(job_id, {})[wave_index] = jpeg_bytes
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check — confirms the server is up and whether the pipeline loaded."""
    return HealthResponse(status="ok", pipeline_ready=_pipeline_ready)

# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_jpeg(video_path: str, timestamp_seconds: float) -> Optional[bytes]:
    """Seek to `timestamp_seconds` in the video and return the frame as JPEG bytes."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        target_frame = int(timestamp_seconds * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None
    finally:
        cap.release()
