import os
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from models import ProcessResponse, StatusResponse, WaveMetadata, WavesResponse
from pipeline import run_pipeline
from wave_detector import detect_waves

app = FastAPI(title="SurfTrak Studio Backend")
_executor = ThreadPoolExecutor(max_workers=2)

# In-memory job store
jobs: dict = {}


def _process_job(job_id: str) -> None:
    job = jobs[job_id]
    try:
        run_pipeline(job["input_path"], job["output_path"], jobs, job_id)

        waves_raw = detect_waves(job["output_path"])
        wave_meta = []
        for i, (start, end) in enumerate(waves_raw):
            wave_meta.append({
                "index": i,
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "duration_seconds": round(end - start, 3),
                "thumbnail_offset": round((start + end) / 2.0, 3),
            })

        best_wave_index = None
        if wave_meta:
            best_wave_index = max(range(len(wave_meta)), key=lambda i: wave_meta[i]["duration_seconds"])

        job["waves"] = wave_meta
        job["wave_count"] = len(wave_meta)
        job["best_wave_index"] = best_wave_index
        job["status"] = "complete"
        job["progress"] = 1.0
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)


@app.post("/process", response_model=ProcessResponse, status_code=202)
async def process_video(background_tasks: BackgroundTasks, video: UploadFile = File(...)):
    if not video.filename or not video.filename.lower().endswith((".mp4", ".mov")):
        raise HTTPException(status_code=400, detail="Only .mp4 and .mov files are accepted")

    job_id = str(uuid.uuid4())
    input_path = f"/tmp/{job_id}_input.mp4"
    output_path = f"/tmp/{job_id}_output.mp4"

    contents = await video.read()
    with open(input_path, "wb") as f:
        f.write(contents)

    jobs[job_id] = {
        "status": "processing",
        "progress": 0.0,
        "input_path": input_path,
        "output_path": output_path,
        "wave_count": None,
        "best_wave_index": None,
        "waves": [],
    }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _process_job, job_id)

    return ProcessResponse(job_id=job_id, status="processing")


@app.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        wave_count=job["wave_count"],
        best_wave_index=job["best_wave_index"],
    )


@app.get("/download/{job_id}")
async def download_video(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "complete":
        raise HTTPException(status_code=404, detail="Output not ready yet")
    if not os.path.exists(job["output_path"]):
        raise HTTPException(status_code=404, detail="Output file missing")

    return FileResponse(
        job["output_path"],
        media_type="video/mp4",
        filename=f"{job_id}_framed.mp4",
    )


@app.get("/waves/{job_id}", response_model=WavesResponse)
async def get_waves(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "complete":
        raise HTTPException(status_code=404, detail="Processing not complete")

    waves = [WaveMetadata(**w) for w in job["waves"]]
    return WavesResponse(waves=waves)


@app.get("/health")
async def health():
    return {"status": "ok"}
