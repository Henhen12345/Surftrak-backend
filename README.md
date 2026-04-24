# SurfTrak Studio — Backend

FastAPI backend for the SurfTrak Studio iOS app. Accepts raw surf session videos, runs a YOLOv8 + Kalman Filter framing pipeline to produce 9:16 vertical crops, auto-detects individual wave clips, and serves the results back to the app.

---

## Local Development

### Prerequisites

- Python 3.10+
- pip

### Run

```bash
./run.sh
```

This installs all dependencies and starts the server with hot-reload.

- Server: http://localhost:8000
- Interactive API docs: http://localhost:8000/docs

The first time you run the pipeline, `ultralytics` will automatically download the YOLOv8n weights (~6 MB). This is expected.

---

## Deploy to Railway

1. Create a free account at [railway.app](https://railway.app)
2. Click **New Project** → **Deploy from GitHub repo**
3. Connect this repository
4. Railway auto-detects the `Procfile` and deploys automatically
5. Once deployed, copy the generated public URL
6. Paste that URL into the iOS app as `BASE_URL`

No build commands or environment variables are needed — Railway handles everything via the `Procfile`.

---

## Environment

No environment variables are required for v1. All tuning parameters are named constants at the top of each source file:

| File | Key constants |
|---|---|
| `pipeline.py` | `LEAD_FACTOR`, `CROP_CENTER_ALPHA`, `ZOOM_ALPHA`, `TARGET_SUBJECT_HEIGHT_RATIO` |
| `wave_detector.py` | `WAVE_VELOCITY_THRESHOLD`, `MIN_WAVE_DURATION_SECONDS`, `WAVE_PAD_SECONDS`, `WAVE_GAP_SECONDS` |

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/process` | Upload a raw `.mp4` or `.mov` video; returns a `job_id` immediately |
| `GET` | `/status/{job_id}` | Poll processing status and progress (0.0–1.0) |
| `GET` | `/download/{job_id}` | Stream the framed 9:16 output video (available when `status = complete`) |
| `GET` | `/waves/{job_id}` | Get metadata for each auto-detected wave clip |
| `GET` | `/health` | Health check — returns `{ "status": "ok" }` |

---

## Architecture Notes

- Jobs are stored in-memory for the lifetime of the server process (suitable for v1 / single-instance Railway deploy).
- CV pipeline and wave detection run in a `ThreadPoolExecutor` (OpenCV and YOLOv8 are not async-native).
- Input and output videos are written to `/tmp/{job_id}_input.mp4` and `/tmp/{job_id}_output.mp4`. Railway's ephemeral filesystem cleans these up on redeploy.
