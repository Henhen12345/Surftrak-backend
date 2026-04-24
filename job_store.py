import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional


class JobStore:
    """Thread-safe in-memory job state manager."""

    def __init__(self) -> None:
        self._store: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def create_job(
        self,
        job_id: str,
        input_video_path: str,
        uwb_log_path: Optional[str],
        output_video_path: str,
    ) -> dict:
        job = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "stage": "Queued",
            "input_video_path": input_video_path,
            "uwb_log_path": uwb_log_path,
            "output_video_path": output_video_path,
            "wave_count": None,
            "best_wave_index": None,
            "duration_seconds": None,
            "waves": [],
            "error_message": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
        }
        with self._lock:
            self._store[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[dict]:
        with self._lock:
            return self._store.get(job_id)

    def update_progress(self, job_id: str, progress: float, stage: str) -> None:
        with self._lock:
            job = self._store.get(job_id)
            if job:
                job["progress"] = round(max(0.0, min(1.0, progress)), 4)
                job["stage"] = stage
                job["status"] = "processing"

    def mark_complete(
        self,
        job_id: str,
        wave_count: int,
        best_wave_index: Optional[int],
        duration_seconds: float,
        waves: List[dict],
    ) -> None:
        with self._lock:
            job = self._store.get(job_id)
            if job:
                job["status"] = "complete"
                job["progress"] = 1.0
                job["stage"] = "Complete"
                job["wave_count"] = wave_count
                job["best_wave_index"] = best_wave_index
                job["duration_seconds"] = duration_seconds
                job["waves"] = waves
                job["completed_at"] = datetime.now(timezone.utc).isoformat()

    def mark_failed(self, job_id: str, error_message: str) -> None:
        with self._lock:
            job = self._store.get(job_id)
            if job:
                job["status"] = "failed"
                job["stage"] = "Failed"
                job["error_message"] = error_message
                job["completed_at"] = datetime.now(timezone.utc).isoformat()

    def job_exists(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._store
