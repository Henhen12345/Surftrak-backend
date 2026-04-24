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
