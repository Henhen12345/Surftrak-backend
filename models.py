from typing import List, Optional
from pydantic import BaseModel


class ProcessResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str          # queued | processing | complete | failed
    progress: float      # 0.0–1.0
    stage: str           # Human-readable current pipeline stage
    wave_count: Optional[int] = None
    best_wave_index: Optional[int] = None   # 0-based index of longest wave
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None     # Populated when status = "failed"
    created_at: str
    completed_at: Optional[str] = None


class WaveMetadata(BaseModel):
    index: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    thumbnail_offset: float   # Best frame timestamp for thumbnail (30% into wave)
    is_best_wave: bool        # True for the longest detected wave


class WavesResponse(BaseModel):
    job_id: str
    waves: List[WaveMetadata]
    total_session_duration: float
    total_ride_time: float    # Sum of all wave durations


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    pipeline_ready: bool      # True if MediaPipe loaded successfully on startup
