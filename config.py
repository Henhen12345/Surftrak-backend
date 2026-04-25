# ── UWB → frame projection ─────────────────────────────────────────────────
CAMERA_HFOV_DEGREES = 62.2        # IMX519 horizontal field of view
FRAME_WIDTH_PX = 1920             # Input video width
FRAME_HEIGHT_PX = 1080            # Input video height
# Width of the CV search strip as a fraction of frame width (±12.5% around UWB column)
UWB_SEARCH_STRIP_WIDTH_FRACTION = 0.25

# ── Subject detection ───────────────────────────────────────────────────────
MOG2_HISTORY = 200
MOG2_VAR_THRESHOLD = 40
MOG2_DETECT_SHADOWS = False
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5
# Blend of UWB column proximity vs MediaPipe detection confidence in subject scoring
MIDLINE_SCORE_WEIGHT = 0.6
# Coast on last known position for this many frames before giving up lock
DETECTION_FALLBACK_HOLD_FRAMES = 90

# ── Surfer-specific detection filters ──────────────────────────────────────
# Water zone: only consider detections whose center_y is BELOW this fraction
# of frame height. The beach/crowd is above this line. Tune per spot.
# 0.35 means the top 35% of the frame is ignored for subject locking.
WATER_ZONE_TOP_FRACTION = 0.35

# Bounding box size filter: surfer apparent height as fraction of frame height.
# Rejects spectators (too tall) and distant noise (too small).
# At typical SurfTrak distances (15–80 ft), a surfer occupies 4%–35% of frame height.
MIN_SUBJECT_HEIGHT_FRACTION = 0.04
MAX_SUBJECT_HEIGHT_FRACTION = 0.35

# Lateral velocity filter: minimum horizontal pixel velocity (px/frame) a candidate
# must demonstrate to qualify as an active surfer. Spectators fail this.
# Computed over a rolling window of recent detections for this candidate.
MIN_LATERAL_VELOCITY_PX_PER_FRAME = 1.8

# Motion continuity: consecutive qualifying frames required before hard lock.
# Prevents locking onto someone who briefly moves through the search strip.
LOCK_CONFIRMATION_FRAMES = 8

# Once locked, frames of non-qualifying motion before the lock is released.
# Higher = stickier lock (better for wipeouts). Lower = faster re-acquire.
LOCK_BREAK_FRAMES = 45

# Sliding window length for computing per-candidate rolling lateral velocity.
VELOCITY_HISTORY_FRAMES = 12

# ── Kalman filter ───────────────────────────────────────────────────────────
KALMAN_PROCESS_NOISE = 1e-2
KALMAN_MEASUREMENT_NOISE = 1e-1

# ── Crop framing ────────────────────────────────────────────────────────────
OUTPUT_ASPECT_W = 9
OUTPUT_ASPECT_H = 16
# Surfer bounding box should fill this fraction of crop height
TARGET_SUBJECT_HEIGHT_RATIO = 0.40
# Hard floor/ceiling on crop height as fraction of full frame height
MIN_CROP_HEIGHT_FRACTION = 0.30
MAX_CROP_HEIGHT_FRACTION = 0.95
# Lower = smoother camera movement, higher = more responsive
CROP_CENTER_ALPHA = 0.08
ZOOM_ALPHA = 0.04
# Crop center leads the surfer in their direction of travel
LEAD_FACTOR = 0.15
# Maximum lead offset as fraction of crop dimension
LEAD_CLAMP = 0.10

# ── Wave detection ──────────────────────────────────────────────────────────
# Kalman velocity magnitude (px/frame) required to qualify as a wave ride
WAVE_VELOCITY_THRESHOLD = 12.0
MIN_WAVE_DURATION_SECONDS = 2.5
WAVE_PAD_SECONDS = 2.0
WAVE_GAP_MERGE_SECONDS = 3.5
WAVE_SAMPLE_FPS = 2  # unused directly — wave detector uses crop_rects list

# ── Processing ──────────────────────────────────────────────────────────────
# Run full detection every N frames; Kalman coasts on skipped frames
PROCESS_EVERY_N_FRAMES = 2
# H.264 quality: 18 = high quality, 23 = default (lower = better / larger file)
OUTPUT_VIDEO_CRF = 18
OUTPUT_VIDEO_PRESET = "medium"    # x264 speed/compression tradeoff
