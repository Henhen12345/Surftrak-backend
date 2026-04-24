from typing import List, Optional
from pydantic import BaseModel


class ProcessResponse(BaseModel):
    job_id: str
    status: str


class StatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    wave_count: Optional[int] = None
    best_wave_index: Optional[int] = None


class WaveMetadata(BaseModel):
    index: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    thumbnail_offset: float


class WavesResponse(BaseModel):
    waves: List[WaveMetadata]
