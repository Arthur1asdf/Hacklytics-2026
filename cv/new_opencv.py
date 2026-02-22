import cv2
import numpy as np
from mss import mss
import time
import csv
import os
import sys
import struct
from pathlib import Path
from datetime import datetime
from collections import Counter
from enum import Enum
import serial
from new_find_model import load_limb_templates
import threading # <-- ADD THIS
from elevenlabs.client import ElevenLabs # <-- ADD THIS
from elevenlabs import play as elevenlabs_play # <-- ADD THIS
try:
    import winsound
except Exception:
    winsound = None

# ============================================================
# INITIALIZE ARDUINO SERIAL CONNECTION
# ============================================================
arduino_serial = None

# ============================================================
# INTERACTIVE ROI ADJUSTMENT MODE
# Set to True to adjust position/size, False for normal monitoring
# ============================================================
ADJUST_ROI_MODE = False  # Set to True to position the box interactively

# ============================================================
# ADJUST THESE VALUES FOR YOUR SCREEN/RESOLUTION  
# ============================================================
ROI_X = 45
ROI_Y = 35
ROI_WIDTH = 150
ROI_HEIGHT = 365

# ROI adjustment controls
MOVE_STEP = 5       # Pixels to move per keypress
SIZE_STEP = 10      # Pixels to resize per keypress

# VISUALIZATION MODE
SHOW_WINDOW = False
SHOW_ONLY_ZOOM = False
SAVE_DEBUG_IMAGE = True
DEBUG_MODE = False
ENABLE_DATA_LOGGING = True
LOG_DIR = Path(__file__).resolve().parent / "logs"
ENV_FILE = Path(__file__).resolve().parent / ".env"

# TTS alerts (ElevenLabs)
ENABLE_TTS_ALERTS = True
TTS_TRIGGER_PHRASE = "you got shot"
TTS_COOLDOWN_SECONDS = 1.5
TTS_VOICE = os.getenv("ELEVENLABS_VOICE", "Rachel")
TTS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "").strip()
TTS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
COLOR_STATES = {"green", "yellow", "red"}
# ============================================================

# Minimum contour area to avoid noise
MIN_CONTOUR_AREA = 15

# Occlusion handling / temporal stability
MISSING_GRACE_FRAMES = 6
MIN_MODEL_VISIBLE_RATIO = 0.03
ZONE_MIN_COVERAGE_RATIO = 0.045
CORE_LIMBS = {'chest', 'torso'}
ALL_LIMBS = ['head', 'chest', 'torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
CONTOUR_CONF_FALLBACK = 0.55
RED_CONTOUR_CONF_FALLBACK = 0.40
RED_IMMEDIATE_CONF = 0.48
RED_LATCH_FRAMES = 2
TEMPLATE_MASK_MIN_PIXELS_RATIO = 0.03

class LimbState(Enum):
    GREEN = '0'
    YELLOW = '1'
    RED = '2'
    MISSING = '3'
    OCCLUDED = '4'
    UNKNOWN = '5'

class LimbType(Enum):
    LEFT_ARM = '0'
    RIGHT_ARM = '1'
    LEFT_LEG = '2'
    RIGHT_LEG = '3'

class LimbStatus:
    def __init__(self):
        self.left_leg = None
        self.right_leg = None
        self.left_arm = None
        self.right_arm = None
        self.prev_left_arm = None
        self.prev_right_arm = None
        self.prev_left_leg = None
        self.prev_right_leg = None

    def setLimbStatus(self, limb_type: LimbType, limb_state: LimbState):
        if limb_type == LimbType.LEFT_ARM:
            self.prev_left_arm = self.left_arm
            self.left_arm = limb_state
        elif limb_type == LimbType.RIGHT_ARM:
            self.prev_right_arm = self.right_arm
            self.right_arm = limb_state
        elif limb_type == LimbType.LEFT_LEG:
            self.prev_left_leg = self.left_leg
            self.left_leg = limb_state
        elif limb_type == LimbType.RIGHT_LEG:
            self.prev_right_leg = self.right_leg
            self.right_leg = limb_state

    def getLimbStatus(self, limb_type: LimbType):
        if limb_type == LimbType.LEFT_ARM:
            return self.left_arm
        elif limb_type == LimbType.RIGHT_ARM:
            return self.right_arm
        elif limb_type == LimbType.LEFT_LEG:
            return self.left_leg
        elif limb_type == LimbType.RIGHT_LEG:
            return self.right_leg
        else:
            return None
    
    def hasStatusChanged(self, limb_type: LimbType):
        if limb_type == LimbType.LEFT_ARM:
            return self.left_arm != self.prev_left_arm
        elif limb_type == LimbType.RIGHT_ARM:
            return self.right_arm != self.prev_right_arm
        elif limb_type == LimbType.LEFT_LEG:
            return self.left_leg != self.prev_left_leg
        elif limb_type == LimbType.RIGHT_LEG:
            return self.right_leg != self.prev_right_leg
        else:
            return False

def _state_from_percentages(red_pct, yellow_pct, green_pct, black_pct, thin_outline=False):
    """Determine state from color percentages. Use lower thresholds for thin-outline sampling."""
    # For thin outline we may have very few pixels; any dominant color wins
    hi = 0.15 if not thin_outline else 0.10
    lo = 0.05 if not thin_outline else 0.02
    black_hi = 0.30 if not thin_outline else 0.25
    black_lo = 0.10 if not thin_outline else 0.05
    state = None
    if red_pct > hi:
        state = 'red'
    elif yellow_pct > hi:
        state = 'yellow'
    elif black_pct > black_hi:
        state = 'missing'
    elif green_pct > hi:
        state = 'green'
    else:
        max_pct = max(red_pct, yellow_pct, green_pct, black_pct)
        if max_pct == red_pct and red_pct > lo:
            state = 'red'
        elif max_pct == yellow_pct and yellow_pct > lo:
            state = 'yellow'
        elif max_pct == black_pct and black_pct > black_lo:
            state = 'missing'
        elif max_pct == green_pct and green_pct > lo:
            state = 'green'
        else:
            state = 'unknown'
    return state


def analyze_color_state(roi_section, mask=None, min_colored_pixels=3, profile='strict'):
    """
    Detect outline color: green (healthy), yellow (warning), red (critical), black (missing)
    Only analyzes pixels along the thin outline, not the body fill.
    Returns dict with state/confidence/coverage metrics.
    """
    if roi_section.size == 0:
        return {
            'state': 'unknown',
            'confidence': 0.0,
            'coverage_ratio': 0.0,
            'red_pct': 0.0,
            'yellow_pct': 0.0,
            'green_pct': 0.0,
            'black_pct': 0.0,
            'total_colored': 0,
        }
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(roi_section, cv2.COLOR_BGR2HSV)
    
    # If mask is provided, only analyze masked pixels (the outline)
    if mask is not None:
        hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
    
    # Define color ranges for the thin HUD outline (match outline mask: neon/wireframe)
    green_mask = cv2.inRange(hsv, np.array([32, 50, 40]), np.array([98, 255, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([15, 50, 40]), np.array([45, 255, 255]))
    red_mask1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([25, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([155, 50, 40]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 38]))
    
    # Count pixels for each color. If we used a mask, only count inside the mask!
    # (Otherwise zeroed-out pixels outside the mask get counted as black and dominate.)
    if mask is not None:
        green_pixels = cv2.countNonZero(cv2.bitwise_and(green_mask, mask))
        yellow_pixels = cv2.countNonZero(cv2.bitwise_and(yellow_mask, mask))
        red_pixels = cv2.countNonZero(cv2.bitwise_and(red_mask, mask))
        black_pixels = cv2.countNonZero(cv2.bitwise_and(black_mask, mask))
        total_pixels = max(int(cv2.countNonZero(mask)), 1)
    else:
        green_pixels = cv2.countNonZero(green_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        red_pixels = cv2.countNonZero(red_mask)
        black_pixels = cv2.countNonZero(black_mask)
        total_pixels = max(int(roi_section.shape[0] * roi_section.shape[1]), 1)
    
    total_colored = green_pixels + yellow_pixels + red_pixels + black_pixels
    coverage_ratio = total_colored / float(total_pixels)

    # Need minimum pixels to make a determination (allow 1 pixel when sampling thin outline)
    if total_colored < min_colored_pixels:
        # If we have at least 1 colored pixel, still try to infer state from it
        if total_colored == 0:
            return {
                'state': 'unknown',
                'confidence': 0.0,
                'coverage_ratio': coverage_ratio,
                'red_pct': 0.0,
                'yellow_pct': 0.0,
                'green_pct': 0.0,
                'black_pct': 0.0,
                'total_colored': total_colored,
            }
        # total_colored >= 1 but < min_colored_pixels: use lower bar for thin outlines
        total = max(total_colored, 1)
        red_pct = red_pixels / total
        yellow_pct = yellow_pixels / total
        green_pct = green_pixels / total
        black_pct = black_pixels / total
        state = _state_from_percentages(red_pct, yellow_pct, green_pct, black_pct, thin_outline=True)
        confidence = min(0.6, 0.3 + 0.1 * total_colored)
        return {
            'state': state,
            'confidence': confidence,
            'coverage_ratio': coverage_ratio,
            'red_pct': red_pct,
            'yellow_pct': yellow_pct,
            'green_pct': green_pct,
            'black_pct': black_pct,
            'total_colored': total_colored,
        }

    # Determine dominant color
    total = max(total_colored, 1)
    red_pct = red_pixels / total
    yellow_pct = yellow_pixels / total
    green_pct = green_pixels / total
    black_pct = black_pixels / total

    state = _state_from_percentages(red_pct, yellow_pct, green_pct, black_pct, thin_outline=False)

    dominant_pct = max(red_pct, yellow_pct, green_pct, black_pct)
    pixel_strength = min(1.0, total_colored / max(float(min_colored_pixels * 2), 1.0))
    confidence = float(np.clip((0.65 * dominant_pct) + (0.35 * pixel_strength), 0.0, 1.0))

    return {
        'state': state,
        'confidence': confidence,
        'coverage_ratio': coverage_ratio,
        'red_pct': red_pct,
        'yellow_pct': yellow_pct,
        'green_pct': green_pct,
        'black_pct': black_pct,
        'total_colored': total_colored,
    }

def initialize_arduino():
    global arduino_serial
    try:
        arduino_serial = serial.Serial(port='/dev/cu.usbmodem101', baudrate=9600, timeout=1)
        print(f"✓ Connected to Arduino on '/dev/cu.usbmodem101' at 9600 baud.")
        time.sleep(2)
    except Exception as e:
        print(f"⚠️  Failed to connect to Arduino: {e}")
arduino_serial = None
tts_client = None
last_tts_time = 0.0
resolved_tts_voice_id = None

def load_local_env(env_path: Path):
    """Load KEY=VALUE pairs from a local .env file into os.environ."""
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            os.environ.setdefault(key, value)
    except Exception as e:
        print(f"Warning: failed to load {env_path.name}: {e}")

# Load .env values for API keys/config before runtime initialization.
load_local_env(ENV_FILE)

def write_arduino(limb_type: LimbType, limb_status: LimbState):
    global arduino_serial

    if arduino_serial is None:
        print("⚠️  Arduino not connected")
        return

    try:
        limb = limb_type.value
        status = limb_status.value
        arduino_serial.write(limb.encode())
        arduino_serial.write(status.encode())
        arduino_serial.flush()
        print(f"Sent to Arduino: Limb={limb}, Status={status}")
        
    except Exception as e:
        print(f"⚠️  Failed to send data to Arduino: {e}")

def init_tts():
    global tts_client, resolved_tts_voice_id
    if not ENABLE_TTS_ALERTS:
        return
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("TTS disabled: ELEVENLABS_API_KEY not set.")
        return
    try:
        tts_client = ElevenLabs(api_key=api_key)
        resolved_tts_voice_id = resolve_voice_id()
        if resolved_tts_voice_id is None:
            print("TTS disabled: unable to resolve ElevenLabs voice ID.")
            tts_client = None
            return
        print(f"TTS enabled (ElevenLabs), voice_id={resolved_tts_voice_id}.")
    except Exception as e:
        print(f"TTS init failed: {e}")
        tts_client = None

def resolve_voice_id():
    """
    Resolve a voice ID for current ElevenLabs SDK versions.
    Priority:
      1) ELEVENLABS_VOICE_ID env var
      2) If ELEVENLABS_VOICE already looks like an ID, use it
      3) Look up by voice name via API
      4) Fallback Rachel public ID
    """
    if tts_client is None:
        return None
    if TTS_VOICE_ID:
        return TTS_VOICE_ID
    if len(TTS_VOICE) >= 20 and " " not in TTS_VOICE:
        return TTS_VOICE
    try:
        voices = tts_client.voices.search(search=TTS_VOICE, page_size=25)
        for v in getattr(voices, "voices", []):
            if str(getattr(v, "name", "")).strip().lower() == TTS_VOICE.strip().lower():
                return getattr(v, "voice_id", None)
        for v in getattr(voices, "voices", []):
            voice_id = getattr(v, "voice_id", None)
            if voice_id:
                return voice_id
    except Exception as e:
        print(f"TTS voice lookup failed: {e}")
    if TTS_VOICE.strip().lower() == "rachel":
        return "21m00Tcm4TlvDq8ikWAM"
    return None

def finalize_wav_header(audio_bytes: bytes) -> bytes:
    """
    ElevenLabs streaming WAV may use placeholder chunk sizes (0xFFFFFFFF).
    Replace RIFF/data chunk sizes with actual values for strict players (winsound).
    """
    if len(audio_bytes) < 44 or audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
        return audio_bytes

    fixed = bytearray(audio_bytes)
    riff_size = len(fixed) - 8
    fixed[4:8] = struct.pack("<I", max(0, riff_size))

    data_idx = bytes(fixed).find(b"data")
    if data_idx != -1 and data_idx + 8 <= len(fixed):
        data_size = len(fixed) - (data_idx + 8)
        fixed[data_idx + 4:data_idx + 8] = struct.pack("<I", max(0, data_size))

    return bytes(fixed)

def _speak_shot_alert():
    if tts_client is None or resolved_tts_voice_id is None:
        return
    try:
        print(f"TTS: {TTS_TRIGGER_PHRASE}")
        audio_stream = tts_client.text_to_speech.convert(
            voice_id=resolved_tts_voice_id,
            text=TTS_TRIGGER_PHRASE,
            model_id=TTS_MODEL,
            output_format="wav_22050",
        )
        audio = finalize_wav_header(b"".join(audio_stream))
        print("TTS: audio generated, playing...")
        if sys.platform.startswith("win") and winsound is not None:
            winsound.PlaySound(audio, winsound.SND_MEMORY)
        elif callable(elevenlabs_play):
            elevenlabs_play(audio)
        elif hasattr(elevenlabs_play, "play"):
            elevenlabs_play.play(audio)
        else:
            raise RuntimeError("Unsupported elevenlabs.play API in installed version.")
        print("TTS: playback finished.")
    except Exception as e:
        print(f"TTS playback failed: {e}")

def trigger_shot_alert():
    global last_tts_time
    if tts_client is None:
        return
    now = time.monotonic()
    if now - last_tts_time < TTS_COOLDOWN_SECONDS:
        return
    last_tts_time = now
    _speak_shot_alert()

def test_tts_once():
    """Quick local test for ElevenLabs TTS without starting vision loop."""
    init_tts()
    if tts_client is None:
        print("TTS test failed: client not initialized.")
        return 1
    print("Running TTS test...")
    trigger_shot_alert()
    print("TTS test complete.")
    return 0

def init_run_logger():
    """
    Initialize CSV logging for downstream data analysis.
    Returns (file_handle, csv_writer, run_id).
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"limb_run_{run_id}.csv"

    fieldnames = [
        "timestamp",
        "run_id",
        "frame_idx",
        "elapsed_s",
        "model_visible_ratio",
        "green_count",
        "yellow_count",
        "red_count",
        "missing_count",
        "occluded_count",
        "unknown_count",
        "critical_load",
    ]
    for limb in ALL_LIMBS:
        fieldnames.append(f"{limb}_state")
        fieldnames.append(f"{limb}_conf")

    fh = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    print(f"Data logging enabled: {log_path}")
    return fh, writer, run_id

def log_frame(writer, run_id, frame_idx, start_time, model_visible_ratio, stable_states, stable_confidence):
    """Write one frame row for analytics."""
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "frame_idx": frame_idx,
        "elapsed_s": round(time.time() - start_time, 2),
        "model_visible_ratio": round(float(model_visible_ratio), 5),
    }

    state_counts = Counter()
    for limb in ALL_LIMBS:
        state = stable_states.get(limb, "unknown")
        conf = float(stable_confidence.get(limb, 0.0))
        row[f"{limb}_state"] = state
        row[f"{limb}_conf"] = round(conf, 4)
        state_counts[state] += 1

    row["green_count"] = state_counts.get("green", 0)
    row["yellow_count"] = state_counts.get("yellow", 0)
    row["red_count"] = state_counts.get("red", 0)
    row["missing_count"] = state_counts.get("missing", 0)
    row["occluded_count"] = state_counts.get("occluded", 0)
    row["unknown_count"] = state_counts.get("unknown", 0)
    row["critical_load"] = (2 * row["red_count"]) + row["yellow_count"]
    writer.writerow(row)

def adjust_roi_interactive():
    """
    Interactive mode to position and scale the ROI box
    
    Controls:
      Arrow Keys: Move the box
      +/= : Make box BIGGER
      -/_ : Make box SMALLER
      W/S : Make box TALLER/SHORTER
      A/D : Make box WIDER/NARROWER
      SPACE : Save and continue to monitoring
      Q/ESC : Quit
    """
    global ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT
    
    print("\n" + "=" * 60)
    print("INTERACTIVE ROI ADJUSTMENT MODE")
    print("=" * 60)
    print("\nControls:")
    print("  Arrow Keys : Move the box")
    print("  + / -      : Scale bigger/smaller")
    print("  W / S      : Make taller/shorter")
    print("  A / D      : Make wider/narrower")
    print("  SPACE      : Save and start monitoring")
    print("  Q / ESC    : Quit")
    print("\n" + "=" * 60)
    
    sct = mss()
    monitor_info = sct.monitors[1]
    monitor_width = monitor_info['width']
    monitor_height = monitor_info['height']
    
    x, y, w, h = ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT
    
    window_name = "ROI Adjuster - Position over player model"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    display_scale = 0.5 if monitor_width > 1920 else 0.7
    display_width = int(monitor_width * display_scale)
    display_height = int(monitor_height * display_scale)
    
    while True:
        screenshot = sct.grab(monitor_info)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Draw overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), -1)
        frame_with_overlay = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        cv2.rectangle(frame_with_overlay, (x, y), (x + w, y + h), (0, 255, 255), 3)
        
        # Crosshair
        center_x, center_y = x + w // 2, y + h // 2
        cv2.line(frame_with_overlay, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
        cv2.line(frame_with_overlay, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)
        
        # Info text
        cv2.putText(frame_with_overlay, f"X:{x} Y:{y} W:{w} H:{h}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Instructions
        inst_y = frame.shape[0] - 120
        cv2.rectangle(frame_with_overlay, (5, inst_y - 20), (400, frame.shape[0] - 5), (0, 0, 0), -1)
        cv2.putText(frame_with_overlay, "Arrows:Move +/-:Scale W/S/A/D:Size", (10, inst_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame_with_overlay, "SPACE:Save & Start | Q:Quit", (10, inst_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        display_frame = cv2.resize(frame_with_overlay, (display_width, display_height))
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(100) & 0xFF
        
        # Movement
        if key == 82 or key == 0:  # Up
            y = max(0, y - MOVE_STEP)
        elif key == 84 or key == 1:  # Down
            y = min(monitor_height - h, y + MOVE_STEP)
        elif key == 81 or key == 2:  # Left
            x = max(0, x - MOVE_STEP)
        elif key == 83 or key == 3:  # Right
            x = min(monitor_width - w, x + MOVE_STEP)
        
        # Scaling
        elif key == ord('+') or key == ord('='):
            w = min(monitor_width - x, w + SIZE_STEP)
            h = min(monitor_height - y, h + SIZE_STEP)
        elif key == ord('-') or key == ord('_'):
            w, h = max(50, w - SIZE_STEP), max(50, h - SIZE_STEP)
        
        # Dimension adjustment
        elif key == ord('w') or key == ord('W'):
            h = min(monitor_height - y, h + SIZE_STEP)
        elif key == ord('s') or key == ord('S'):
            h = max(50, h - SIZE_STEP)
        elif key == ord('d') or key == ord('D'):
            w = min(monitor_width - x, w + SIZE_STEP)
        elif key == ord('a') or key == ord('A'):
            w = max(50, w - SIZE_STEP)
        
        # Save
        elif key == ord(' '):
            ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT = x, y, w, h
            cv2.destroyAllWindows()
            print(f"\n✓ ROI Updated: X={x}, Y={y}, Width={w}, Height={h}")
            
            with open('roi_settings.txt', 'w') as f:
                f.write(f"ROI_X = {x}\nROI_Y = {y}\nROI_WIDTH = {w}\nROI_HEIGHT = {h}\n")
            print("✓ Saved to roi_settings.txt")
            
            if w != ROI_WIDTH or h != ROI_HEIGHT:
                print("\n⚠️  Size changed! Update new_find_model.py:")
                print(f"   TARGET_WIDTH = {w}")
                print(f"   TARGET_HEIGHT = {h}")
                print("   Then run: python new_find_model.py\n")
            return True
        
        # Quit
        elif key == ord('q') or key == ord('Q') or key == 27:
            cv2.destroyAllWindows()
            print("\n❌ Cancelled")
            return False
    
    return False

def main():
    """
    Main function to monitor player model limb colors using contour detection
    """
    # Check if we should run ROI adjustment mode first
    if ADJUST_ROI_MODE:
        if not adjust_roi_interactive():
            return  # User cancelled
    
    initialize_arduino()
    init_tts()
    
    # Initialize data logger for analytics
    log_fh = None
    log_writer = None
    run_id = None
    frame_idx = 0
    start_time = time.time()
    if ENABLE_DATA_LOGGING:
        log_fh, log_writer, run_id = init_run_logger()

    print("=" * 60)
    print("Starting Player Health Monitor...")
    print("=" * 60)
    if SHOW_WINDOW:
        print("Press 'q' to quit")
        print("\nVISUALIZATION MODE:")
        if SHOW_ONLY_ZOOM:
            print("- Showing ONLY zoomed view (SHOW_ONLY_ZOOM = True)")
            print("- You should see the player model in the window")
            print("- If not, adjust ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT")
        else:
            print("1. Look at the window that appears")
            print("2. Find where the player model is on your screen")
            print("3. Edit the code to adjust these values:")
        print(f"   - ROI_X (currently {ROI_X}) - Move box LEFT/RIGHT")
        print(f"   - ROI_Y (currently {ROI_Y}) - Move box UP/DOWN")
        print(f"   - ROI_WIDTH (currently {ROI_WIDTH}) - Make box WIDER/NARROWER")
        print(f"   - ROI_HEIGHT (currently {ROI_HEIGHT}) - Make box TALLER/SHORTER")
        print("\nThe box should capture the entire player model!")
    else:
        print("Press Ctrl+C to quit")
    print(f"\nCurrent detection box: Position ({ROI_X}, {ROI_Y}), Size {ROI_WIDTH}x{ROI_HEIGHT}")
    print("=" * 60)
    print("Monitoring started...\n")
    
    # Initialize screen capture
    sct = mss()
    
    # Get screen dimensions dynamically
    monitor_info = sct.monitors[1]
    monitor = {"top": 0, "left": 0, "width": monitor_info['width'], "height": monitor_info['height']}
    
    print(f"Screen resolution: {monitor['width']}x{monitor['height']}")
    
    # Load custom templates from reference image
    print("\nLoading limb templates...")
    template_masks = load_limb_templates()
    
    if template_masks is None:
        print("❌ Failed to load templates!")
        print("   Run: python new_find_model.py")
        return
    
    print("✓ Templates loaded successfully!\n")
    
    # Track all expected limbs
    limbs_ever_seen = set()
    
    # Temporal smoothing
    detection_history = {}
    HISTORY_SIZE = 4
    DAMAGE_CONFIDENCE = 2
    SAFE_CONFIDENCE = 3
    consecutive_absent = {limb: 0 for limb in ALL_LIMBS}
    red_latch_remaining = {limb: 0 for limb in ALL_LIMBS}
    
    debug_image_saved = False
    limb_object = LimbStatus()
    prev_color_states = {}
    
    try:
        while True:
            # Capture screen
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Extract the region of interest
            roi_frame = frame[ROI_Y:ROI_Y+ROI_HEIGHT, ROI_X:ROI_X+ROI_WIDTH].copy()
            
            # Convert to HSV
            hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
            
            # Build thin HUD outline mask: green + yellow + red (permissive to catch neon)
            green_outline = cv2.inRange(hsv_roi, np.array([32, 50, 40]), np.array([98, 255, 255]))
            yellow_outline = cv2.inRange(hsv_roi, np.array([15, 50, 40]), np.array([45, 255, 255]))
            red_outline1 = cv2.inRange(hsv_roi, np.array([0, 50, 40]), np.array([25, 255, 255]))
            red_outline2 = cv2.inRange(hsv_roi, np.array([155, 50, 40]), np.array([180, 255, 255]))
            red_outline = cv2.bitwise_or(red_outline1, red_outline2)
            thin_outline_mask = cv2.bitwise_or(cv2.bitwise_or(green_outline, yellow_outline), red_outline)
            # Slight dilate so 1–2 px outline overlaps limb template regions
            kernel_dilate = np.ones((2, 2), np.uint8)
            thin_outline_mask = cv2.dilate(thin_outline_mask, kernel_dilate, iterations=1)
            
            # Legacy: green-only mask for contour/visibility (keep for model_visible_ratio)
            green_outline_mask = cv2.inRange(hsv_roi, np.array([40, 100, 80]), np.array([90, 255, 255]))
            edges = cv2.Canny(green_outline_mask, 30, 100)
            kernel = np.ones((2, 2), np.uint8)
            color_mask = cv2.dilate(edges, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            model_visible_ratio = cv2.countNonZero(color_mask) / float(ROI_WIDTH * ROI_HEIGHT)
            
            # Save debug images on first frame
            if SAVE_DEBUG_IMAGE and not debug_image_saved:
                cv2.imwrite('debug_captured_area.png', roi_frame)
                cv2.imwrite('debug_mask.png', color_mask)
                cv2.imwrite('debug_thin_outline.png', thin_outline_mask)
                
                # Save contours visualization with templates
                debug_contours = roi_frame.copy()
                cv2.drawContours(debug_contours, contours, -1, (0, 255, 255), 1)
                
                # Draw template masks on debug image
                for limb_name in ALL_LIMBS:
                    if limb_name in template_masks:
                        contours_mask, _ = cv2.findContours(template_masks[limb_name], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(debug_contours, contours_mask, -1, (255, 200, 0), 1)
                
                cv2.imwrite('debug_contours.png', debug_contours)
                
                print(f"✓ Saved debug images: debug_captured_area.png, debug_mask.png, debug_thin_outline.png, debug_contours.png")
                print(f"  Check these to see what is being detected\n")
                debug_image_saved = True
            
            # Primary traced-mask detection for all limbs
            current_states = {}
            frame_signal_conf = {limb: 0.0 for limb in ALL_LIMBS}
            
            for limb_name in ALL_LIMBS:
                if limb_name not in template_masks:
                    continue
                    
                limb_mask = template_masks[limb_name]
                # Sample only the thin HUD outline within this limb (outline ∩ limb)
                outline_in_limb = cv2.bitwise_and(limb_mask, thin_outline_mask)
                outline_pixels = cv2.countNonZero(outline_in_limb)
                # If no outline detected in this limb, fall back to full limb mask
                use_mask = outline_in_limb if outline_pixels > 0 else limb_mask
                min_pixels = 1 if outline_pixels > 0 else max(2, int(cv2.countNonZero(limb_mask) * TEMPLATE_MASK_MIN_PIXELS_RATIO))
                profile = 'core_relaxed' if limb_name in CORE_LIMBS else 'strict'
                
                limb_analysis = analyze_color_state(
                    roi_frame,
                    mask=use_mask,
                    min_colored_pixels=min_pixels,
                    profile=profile,
                )
                limb_state = limb_analysis['state']
                limb_conf = limb_analysis['confidence']

                if limb_state in ['green', 'yellow', 'red']:
                    current_states[limb_name] = limb_state
                    frame_signal_conf[limb_name] = limb_conf
                    limbs_ever_seen.add(limb_name)
                elif DEBUG_MODE:
                    print(f"  {limb_name}: {limb_state} (conf: {limb_conf:.2f}, colored: {limb_analysis['total_colored']})")
            
            # Update detection history for temporal smoothing
            for limb_name in ALL_LIMBS:
                if limb_name not in detection_history:
                    detection_history[limb_name] = []
                
                if limb_name in current_states:
                    detection_history[limb_name].append(current_states[limb_name])
                    consecutive_absent[limb_name] = 0
                else:
                    if limb_name in limbs_ever_seen:
                        if model_visible_ratio < MIN_MODEL_VISIBLE_RATIO:
                            detection_history[limb_name].append('occluded')
                            frame_signal_conf[limb_name] = max(0.0, 1.0 - (model_visible_ratio / max(MIN_MODEL_VISIBLE_RATIO, 1e-6)))
                        else:
                            consecutive_absent[limb_name] += 1
                            if consecutive_absent[limb_name] >= MISSING_GRACE_FRAMES:
                                detection_history[limb_name].append('missing')
                                frame_signal_conf[limb_name] = min(1.0, consecutive_absent[limb_name] / float(MISSING_GRACE_FRAMES))
                            else:
                                detection_history[limb_name].append('occluded')
                                frame_signal_conf[limb_name] = consecutive_absent[limb_name] / float(MISSING_GRACE_FRAMES)
                    else:
                        detection_history[limb_name].append('unknown')
                
                if len(detection_history[limb_name]) > HISTORY_SIZE:
                    detection_history[limb_name].pop(0)
            
            # Determine stable state for each limb
            stable_states = {}
            stable_confidence = {}
            for limb_name in ALL_LIMBS:
                if limb_name not in detection_history or len(detection_history[limb_name]) == 0:
                    continue
                
                state_counts = Counter(detection_history[limb_name])
                most_common_state, count = state_counts.most_common(1)[0]
                
                if most_common_state in ['red', 'yellow']:
                    required_confidence = DAMAGE_CONFIDENCE
                else:
                    required_confidence = SAFE_CONFIDENCE
                
                if count >= required_confidence:
                    stable_states[limb_name] = most_common_state
                    temporal_conf = count / max(len(detection_history[limb_name]), 1)
                    stable_confidence[limb_name] = (0.5 * temporal_conf) + (0.5 * frame_signal_conf.get(limb_name, 0.0))

            # Immediate red override
            for limb_name in ALL_LIMBS:
                current_state = current_states.get(limb_name)
                current_conf = frame_signal_conf.get(limb_name, 0.0)
                if current_state == 'red' and current_conf >= RED_IMMEDIATE_CONF:
                    stable_states[limb_name] = 'red'
                    stable_confidence[limb_name] = max(stable_confidence.get(limb_name, 0.0), current_conf)
                    red_latch_remaining[limb_name] = RED_LATCH_FRAMES
                elif red_latch_remaining[limb_name] > 0:
                    stable_states[limb_name] = 'red'
                    stable_confidence[limb_name] = max(stable_confidence.get(limb_name, 0.0), 0.45)
                    red_latch_remaining[limb_name] -= 1

            # TTS alert when any stable color changes for a limb.
            color_changed = False
            for limb_name in ALL_LIMBS:
                current_color = stable_states.get(limb_name, "unknown")
                previous_color = prev_color_states.get(limb_name)
                if previous_color in COLOR_STATES and current_color in COLOR_STATES and current_color != previous_color:
                    color_changed = True
                prev_color_states[limb_name] = current_color
            if color_changed:
                trigger_shot_alert()
            
            # Print status
            print("\n--- Frame Update ---")
            
            state_emoji = {
                'red': '🔴',
                'yellow': '🟡',
                'green': '🟢',
                'missing': '❌',
                'occluded': '⚫',
                'unknown': '⚪',
            }
            
            for limb_name in ALL_LIMBS:
                color_state = stable_states.get(limb_name, 'unknown')
                conf = stable_confidence.get(limb_name, 0.0)
                icon = state_emoji.get(color_state, '⚪')
                print(f"{icon}  {limb_name.replace('_', ' ').title()}: {color_state.upper()} (conf: {conf:.2f})")
                
                # SEND STATUS TO ARDUINO
                if limb_name in ["left_arm", "left_leg", "right_arm", "right_leg"]:
                    limb_type = LimbType[limb_name.upper()]
                    limb_status = LimbState[color_state.upper()]
                    limb_object.setLimbStatus(limb_type=limb_type, limb_state=limb_status)

                    if limb_object.hasStatusChanged(limb_type=limb_type):
                        write_arduino(limb_type, limb_status)

            if ENABLE_DATA_LOGGING and log_writer is not None:
                log_frame(
                    writer=log_writer,
                    run_id=run_id,
                    frame_idx=frame_idx,
                    start_time=start_time,
                    model_visible_ratio=model_visible_ratio,
                    stable_states=stable_states,
                    stable_confidence=stable_confidence,
                )
                frame_idx += 1

            # Show visualization window if enabled
            if SHOW_WINDOW:
                if SHOW_ONLY_ZOOM:
                    # Only show the zoomed view - easier to position
                    if roi_frame.size > 0:
                        display_zoom = roi_frame.copy()
                        cv2.putText(
                            display_zoom,
                            f"Pos: ({ROI_X},{ROI_Y}) Size: {ROI_WIDTH}x{ROI_HEIGHT}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                        )
                        cv2.putText(
                            display_zoom,
                            "Adjust ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )
                        cv2.imshow(
                            "What the detector sees - Should show player model",
                            cv2.resize(display_zoom, (600, 825)),
                        )
                else:
                    # Show both windows
                    frame_display = frame.copy()
                    cv2.rectangle(
                        frame_display,
                        (ROI_X, ROI_Y),
                        (ROI_X + ROI_WIDTH, ROI_Y + ROI_HEIGHT),
                        (0, 255, 255),
                        4,
                    )
                    cv2.putText(
                        frame_display,
                        "<-- MOVE THIS YELLOW BOX TO COVER PLAYER MODEL",
                        (ROI_X + ROI_WIDTH + 10, ROI_Y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame_display,
                        f"Position: ({ROI_X},{ROI_Y}) | Size: {ROI_WIDTH}x{ROI_HEIGHT}",
                        (ROI_X, ROI_Y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                    scale = 0.5 if monitor['width'] > 2000 else 0.6
                    cv2.imshow(
                        "YOUR SCREEN - Find the player model and move yellow box to it",
                        cv2.resize(
                            frame_display,
                            (int(monitor['width'] * scale), int(monitor['height'] * scale)),
                        ),
                    )

                    if roi_frame.size > 0:
                        cv2.imshow(
                            "Inside the Yellow Box (what will be analyzed)",
                            cv2.resize(roi_frame, (600, 825)),
                        )

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Small delay
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        print("Goodbye!")
    finally:
        if log_fh is not None:
            log_fh.close()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    if "--test-tts" in sys.argv:
        raise SystemExit(test_tts_once())
    main()
