# SurfTrak Studio Backend

FastAPI backend for the SurfTrak Studio iOS app. Receives raw surf session videos
and UWB ranging logs from the SurfTrak camera unit, runs a hybrid UWB + computer
vision framing pipeline, auto-detects individual wave clips, and serves the results
back to the app.

---

## Architecture

### Why UWB + CV instead of pure CV?

Open-water tracking is hard for pure computer vision: whitewash, crowds of surfers,
severe backlighting at golden hour, and partial occlusion during wipeouts all cause
pure-CV trackers to lose the subject or latch onto the wrong person.

SurfTrak solves this with **sensor fusion**:

- **UWB (Ultra-Wideband) ranging** tells us *exactly* where the surfer is horizontally
  at every moment — even in zero-visibility conditions.
- **CV (MediaPipe Pose + MOG2)** operates only within the narrow strip the UWB predicts.
  It handles precise crop framing, not subject location.

The result: UWB provides a guaranteed horizontal bounding column; CV refines it to
a tight body bounding box used to drive the Kalman crop engine.

```
  iOS App
    │
    ├── POST /process  (video + uwb_log)
    │
    ▼
  UWB Parser ──► predicted_column per frame (pan angle from dual anchors)
    │
    ▼
  Subject Finder  (MOG2 motion mask + MediaPipe Pose, search strip = UWB ± 12.5%)
    │
    ▼
  Crop Engine  (4-state Kalman filter + exponential smoothing + velocity lead room)
    │
    ▼
  Renderer  (H.264 output, 1080×1920, LANCZOS4 resize)
    │
    ▼
  Wave Detector  (velocity threshold on crop-rect time series → wave segments)
    │
    ▼
  GET /waves        — wave metadata list
  GET /download     — framed output video
  GET /thumbnail    — JPEG thumbnail per wave
```

---

## UWB Log Format

The Pi firmware writes a JSON file alongside each session video.
This is the firmware-to-backend contract:

```json
{
  "session_start_unix": 1714000000.0,
  "anchor_baseline_cm": 20.0,
  "readings": [
    {"t": 0.050, "d1_cm": 4823.0, "d2_cm": 4819.0},
    {"t": 0.100, "d1_cm": 4821.0, "d2_cm": 4822.0}
  ]
}
```

| Field | Description |
|---|---|
| `session_start_unix` | Unix timestamp when recording began (informational only) |
| `anchor_baseline_cm` | Distance between the two UWB anchors in centimetres |
| `readings[].t` | Seconds since session start |
| `readings[].d1_cm` | Distance from **left** anchor to surfer beacon (cm) |
| `readings[].d2_cm` | Distance from **right** anchor to surfer beacon (cm) |

**Rules the firmware must follow:**
- Readings must be in ascending time order
- Readings with `d1_cm` or `d2_cm` ≤ 0 are skipped by the backend
- Aim for a reading rate ≥ 20 Hz; gaps > 500 ms cause the backend to hold last known
- The UWB log is **optional** — if omitted, the backend falls back to full-frame CV

---

## Local Development

### Prerequisites

- Python 3.10+
- pip

### Run

```bash
./run.sh
```

- Server: http://localhost:8000
- API docs: http://localhost:8000/docs

**First run:** `ultralytics`/MediaPipe weights download automatically (~50 MB). This is expected.

### Test with curl

```bash
# Upload a video (no UWB log — pure CV fallback)
JOB=$(curl -s -X POST http://localhost:8000/process \
  -F "video=@/path/to/session.mp4" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

echo "Job ID: $JOB"

# Poll until complete
while true; do
  STATUS=$(curl -s http://localhost:8000/status/$JOB)
  echo $STATUS | python3 -m json.tool
  DONE=$(echo $STATUS | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  [ "$DONE" = "complete" ] || [ "$DONE" = "failed" ] && break
  sleep 3
done

# Download framed video
curl -OJ http://localhost:8000/download/$JOB

# Get wave list
curl http://localhost:8000/waves/$JOB | python3 -m json.tool

# Get thumbnail for wave 0
curl http://localhost:8000/thumbnail/$JOB/0 --output wave0_thumb.jpg
```

---

## Deploying to Railway

1. Create a free account at [railway.app](https://railway.app)
2. Click **"New Project"** → **"Deploy from GitHub repo"**
3. Authenticate with GitHub and select this repository
4. Railway auto-detects the `Procfile` — no build command needed
5. Click **"Deploy"** and wait ~2 minutes for the first build
6. Once green, click **"Settings"** → copy the **Public Domain** URL
7. Paste that URL into the iOS app as `BASE_URL`

**Important:** Railway free tier has 512 MB RAM. Sessions over ~30 minutes may hit
this limit. Upgrade to the **Starter plan ($5/month)** for production use.

---

## Tuning the Pipeline

All constants live in `config.py`. No logic code needs to change to tune the pipeline.

### UWB Projection

| Constant | Controls | Tune when |
|---|---|---|
| `CAMERA_HFOV_DEGREES` | Horizontal field of view used to project UWB angle to pixel column | Wrong camera lens or digital crop is active |
| `UWB_SEARCH_STRIP_WIDTH_FRACTION` | Width of the CV search region (default ±12.5% = 25% total) | Increase if UWB is noisy / surfer keeps escaping the strip; decrease to reduce false positives in crowds |

### Subject Detection

| Constant | Controls | Tune when |
|---|---|---|
| `MOG2_VAR_THRESHOLD` | Background subtractor sensitivity (lower = more sensitive) | Too many false motion blobs → increase; missing surfer in flat light → decrease |
| `MOG2_HISTORY` | Background model learning window (frames) | Slow-moving camera → increase; fast scene changes → decrease |
| `MEDIAPIPE_MIN_DETECTION_CONFIDENCE` | Minimum pose confidence to count as a detection | Too many false locks → increase; losing surfer in poor visibility → decrease |
| `MIDLINE_SCORE_WEIGHT` | How much to weight UWB proximity vs pose confidence in candidate scoring | Noisy UWB → decrease towards 0.4; reliable UWB → increase towards 0.8 |
| `DETECTION_FALLBACK_HOLD_FRAMES` | Frames to coast on last known position before releasing lock | Short wipeouts → increase; avoid locking onto static objects → decrease |

### Kalman Filter

| Constant | Controls | Tune when |
|---|---|---|
| `KALMAN_PROCESS_NOISE` | How much the filter trusts the motion model vs measurements | Camera jumpy → increase; filter lags detections → decrease |
| `KALMAN_MEASUREMENT_NOISE` | How much the filter trusts each detection | Noisy bounding boxes → increase; detections are very accurate → decrease |

### Crop Framing

| Constant | Controls | Tune when |
|---|---|---|
| `TARGET_SUBJECT_HEIGHT_RATIO` | Surfer fills this fraction of crop height (default 40%) | Surfer too small in output → decrease; too zoomed in → increase |
| `CROP_CENTER_ALPHA` | Speed of crop center smoothing (0 = frozen, 1 = instant) | Camera feels jittery → decrease; camera lags subject → increase |
| `ZOOM_ALPHA` | Speed of zoom level smoothing | Zoom pumps on whitewash → decrease; slow to respond to close-ups → increase |
| `LEAD_FACTOR` | How far ahead of the surfer the crop leads (direction of travel) | Surfer running off frame edge → increase; lead looks unnatural → decrease |
| `LEAD_CLAMP` | Maximum lead offset as fraction of crop dimension | Cap LEAD_FACTOR effect on very fast waves |
| `MIN_CROP_HEIGHT_FRACTION` | Never crop smaller than this fraction of frame height | Adjust if maximum zoom in feels too tight |
| `MAX_CROP_HEIGHT_FRACTION` | Never crop larger than this fraction of frame height | Adjust to limit zoom-out on slow paddling scenes |

### Wave Detection

| Constant | Controls | Tune when |
|---|---|---|
| `WAVE_VELOCITY_THRESHOLD` | Minimum crop velocity (px/frame) to qualify as riding | Non-riding paddling triggers waves → increase; short fast waves missed → decrease |
| `MIN_WAVE_DURATION_SECONDS` | Minimum ride duration to include a wave | Whitewash bursts flagged as waves → increase |
| `WAVE_PAD_SECONDS` | Seconds added before/after each wave | Pop-up or kickout getting cut off → increase |
| `WAVE_GAP_MERGE_SECONDS` | Merge waves separated by less than this gap | One wave split into two by a brief stall → increase |

### Processing

| Constant | Controls | Tune when |
|---|---|---|
| `PROCESS_EVERY_N_FRAMES` | Run full detection every N frames (Kalman coasts between) | Quality vs speed tradeoff; set to 1 for maximum accuracy |

---

## API Reference

### `POST /process`

Upload a session video and optional UWB log. Returns immediately with a `job_id`.

```bash
curl -X POST https://your-app.railway.app/process \
  -F "video=@session.mp4" \
  -F "uwb_log=@session_uwb.json"
```

Response `202`:
```json
{
  "job_id": "d4e5f6a7-...",
  "status": "queued",
  "message": "Processing started. Poll /status/{job_id} for updates."
}
```

---

### `GET /status/{job_id}`

Poll for processing progress.

```bash
curl https://your-app.railway.app/status/d4e5f6a7-...
```

Response `200`:
```json
{
  "job_id": "d4e5f6a7-...",
  "status": "processing",
  "progress": 0.42,
  "stage": "Detecting subject",
  "wave_count": null,
  "best_wave_index": null,
  "duration_seconds": null,
  "error_message": null,
  "created_at": "2025-04-24T10:00:00+00:00",
  "completed_at": null
}
```

`status` values: `queued` | `processing` | `complete` | `failed`

---

### `GET /download/{job_id}`

Download the framed 9:16 output video (only when `status = complete`).

```bash
curl -OJ https://your-app.railway.app/download/d4e5f6a7-...
```

Response: `video/mp4` stream, 1080×1920, H.264.
Returns `409` if job is not complete.

---

### `GET /waves/{job_id}`

Get metadata for each auto-detected wave clip.

```bash
curl https://your-app.railway.app/waves/d4e5f6a7-...
```

Response `200`:
```json
{
  "job_id": "d4e5f6a7-...",
  "waves": [
    {
      "index": 0,
      "start_seconds": 12.4,
      "end_seconds": 26.1,
      "duration_seconds": 13.7,
      "thumbnail_offset": 16.5,
      "is_best_wave": true
    }
  ],
  "total_session_duration": 180.0,
  "total_ride_time": 13.7
}
```

---

### `GET /thumbnail/{job_id}/{wave_index}`

Return a JPEG thumbnail frame for a specific wave. Cached after first request.

```bash
curl https://your-app.railway.app/thumbnail/d4e5f6a7-.../0 --output wave0.jpg
```

Response: `image/jpeg`, 1080×1920.
Returns `404` if wave index is out of range, `409` if not complete.

---

### `GET /health`

```bash
curl https://your-app.railway.app/health
```

Response `200`:
```json
{"status": "ok", "version": "1.0.0", "pipeline_ready": true}
```

`pipeline_ready` is `false` if MediaPipe failed to load at startup.

---

## Known Limitations (v1)

- **In-memory job store** resets on every server restart or Railway redeploy.
  Re-upload sessions after a deploy if needed.
- **Max 2 concurrent jobs** enforced to stay within Railway free tier memory budget.
  Excess requests receive `503 Service Unavailable`.
- **UWB log is optional** but strongly recommended. Pure CV fallback (`uwb_log`
  omitted) is less reliable in crowded lineups or poor lighting.
- **Long sessions** (3+ hours continuous) may exceed Railway free tier's 512 MB RAM.
  Upgrade to Railway Starter ($5/month) for production use with full-length sessions.
