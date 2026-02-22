import cv2
import numpy as np
from mss import mss
import time
from collections import Counter
from enum import Enum
import serial

# ============================================================
# ADJUST THESE VALUES FOR YOUR SCREEN/RESOLUTION  
# ============================================================
# Based on your screenshot at 2880x1864 resolution
# Player model is at top-left of YouTube video

ROI_X = 17       # Left edge where player model appears
ROI_Y = 35         # Below browser tabs, at video content
ROI_WIDTH = 200     # Width of player model
ROI_HEIGHT = 370    # Height of player model (head to feet)

# VISUALIZATION MODE
SHOW_WINDOW = False  # No GUI - runs in background
SHOW_ONLY_ZOOM = False
SAVE_DEBUG_IMAGE = True  # Saves one image to verify position
DEBUG_MODE = False  # Print all detections for debugging
# ============================================================

# Minimum contour area to avoid noise (adjust if getting too many/few detections)
MIN_CONTOUR_AREA = 15  # Very low to catch all limb sections

# Occlusion handling / temporal stability
MISSING_GRACE_FRAMES = 6  # Require several consecutive misses before "missing"
MIN_MODEL_VISIBLE_RATIO = 0.03  # If too little model color is visible, treat as occluded
ZONE_MIN_COVERAGE_RATIO = 0.045  # Fallback zone must have enough colored pixels to trust state
CORE_LIMBS = {'chest', 'torso'}  # Only these get relaxed dark-red handling
ALL_LIMBS = ['head', 'chest', 'torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
CONTOUR_CONF_FALLBACK = 0.55  # Generic contour fallback threshold
RED_CONTOUR_CONF_FALLBACK = 0.40  # Allow red through with lower contour confidence
RED_IMMEDIATE_CONF = 0.48  # If current-frame red confidence exceeds this, show red immediately
RED_LATCH_FRAMES = 2  # Keep immediate red visible briefly to avoid flicker
TEMPLATE_MASK_MIN_PIXELS_RATIO = 0.03  # Required colored coverage within each traced limb mask (lower for thin limbs)

# Small pixel offsets to shift the traced/template limb masks when they
# don't line up perfectly with your captured contours. Adjust these to
# move all template boxes at once (can be positive or negative).
TEMPLATE_SHIFT_X = 2
TEMPLATE_SHIFT_Y = 2

# Per-limb fine alignment (px). Tweak these to position each template independently.
# Format: 'limb': (x_offset_pixels, y_offset_pixels)
PER_LIMB_OFFSETS = {
    'head': (0, -8),
    'chest': (0, -30),
    'torso': (0, -38),
    'left_arm': (-8, 4),
    'right_arm': (8, 4),
    'left_leg': (-6, -20),
    'right_leg': (6, -20),
}

class LimbState(Enum):
    GREEN = '0',
    YELLOW = '1',
    RED = '2',
    MISSING = '3',
    OCCLUDED = '4',
    UNKNOWN = '5',

class LimbType(Enum):
    LEFT_ARM = '0',
    RIGHT_ARM = '1',
    LEFT_LEG = '2',
    RIGHT_LEG = '3',

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

# How many pixels to shrink the chest polygon by (approx). Increase to reduce overlap.
CHEST_SHRINK_PX = 8
# How many pixels to shrink the torso polygon by (approx). Increase to reduce overlap.
TORSO_SHRINK_PX = 26

# Leg growth (px) to make leg boxes larger and fit the silhouette better
LEG_GROW_PX = 10
# Additional pixels to add to leg stroke thickness
LEG_THICK_ADJUST = 2
# Fallback limb zones (x1, y1, x2, y2) normalized to ROI dimensions
LIMB_ZONES = {
    'head': (0.30, 0.00, 0.70, 0.25),
    'chest': (0.30, 0.20, 0.70, 0.40),
    'torso': (0.30, 0.40, 0.70, 0.60),
    # Keep arm zones tighter/outside center body so missing-arm fallback is less prone to torso bleed.
    'left_arm': (0.00, 0.15, 0.22, 0.60),
    'right_arm': (0.78, 0.15, 1.00, 0.60),
    'left_leg': (0.20, 0.60, 0.50, 1.00),
    'right_leg': (0.50, 0.60, 0.80, 1.00),
}

def identify_limb(contour, roi_width, roi_height):
    """
    Identify which limb a contour represents based on its position and shape
    Returns: limb name string
    """
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return "unknown"
    
    # Get centroid
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Normalize coordinates (0-1)
    cx_norm = cx / roi_width
    cy_norm = cy / roi_height
    
    # Get contour dimensions
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = h / w if w > 0 else 0
    
    # Identify based on position with clear non-overlapping zones
    # Use the bounding box to get better position info
    x, y, w, h = cv2.boundingRect(contour)
    y_top = y / roi_height  # Top of the contour
    y_bottom = (y + h) / roi_height  # Bottom of the contour
    y_center = cy_norm
    
    # Head: very top area (top of contour < 0.25)
    if y_top < 0.25 and 0.3 < cx_norm < 0.7:
        return "head"
    
    # Chest area (center 0.2-0.4) - upper torso, center position
    if 0.2 <= y_center < 0.4 and 0.3 < cx_norm < 0.7:
        return "chest"
    
    # Torso/abdomen area (center 0.4-0.6) - lower torso, center position
    if 0.4 <= y_center < 0.6 and 0.3 < cx_norm < 0.7:
        return "torso"
    
    # Arms: on the sides, anywhere from top to mid-body
    if y_center < 0.6:
        if cx_norm <= 0.3:
            return "left_arm"
        elif cx_norm >= 0.7:
            return "right_arm"
    
    # Legs (center >= 0.6)
    if y_center >= 0.6:
        if cx_norm < 0.5:
            return "left_leg"
        else:
            return "right_leg"
    
    return "unknown"

def analyze_color_state(roi_section, mask=None, min_colored_pixels=3, profile='strict'):
    """
    Detect if the limb is green (healthy), yellow (warning), or red (critical)
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
            'total_colored': 0,
        }
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(roi_section, cv2.COLOR_BGR2HSV)
    
    # If mask is provided, only analyze masked pixels
    if mask is not None:
        hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
    
    # Define color ranges for classification
    # All require minimum saturation (80) to avoid environmental interference
    # Green: hue 40-90
    green_mask = cv2.inRange(hsv, np.array([40, 80, 38]), np.array([90, 255, 255]))
    
    # Yellow: hue 15-45
    yellow_mask = cv2.inRange(hsv, np.array([15, 80, 38]), np.array([45, 255, 255]))
    
    # Red: hue 0-15 and 160-180
    red_mask1 = cv2.inRange(hsv, np.array([0, 80, 38]), np.array([15, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([160, 80, 38]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Count pixels for each color
    green_pixels = cv2.countNonZero(green_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    red_pixels = cv2.countNonZero(red_mask)
    
    total_colored = green_pixels + yellow_pixels + red_pixels

    # If a mask was provided, compute coverage relative to the masked area
    if mask is not None:
        total_pixels = max(int(cv2.countNonZero(mask)), 1)
    else:
        total_pixels = max(int(roi_section.shape[0] * roi_section.shape[1]), 1)
    coverage_ratio = total_colored / float(total_pixels)

    # Need minimum pixels to make a determination
    if total_colored < min_colored_pixels:
        return {
            'state': 'unknown',
            'confidence': 0.0,
            'coverage_ratio': coverage_ratio,
            'red_pct': 0.0,
            'yellow_pct': 0.0,
            'green_pct': 0.0,
            'total_colored': total_colored,
        }
    
    # Determine dominant color with clear priority
    # Check each color independently using colored pixels as the reference
    total = max(total_colored, 1)

    red_pct = red_pixels / total
    yellow_pct = yellow_pixels / total
    green_pct = green_pixels / total

    # Determine if we should be lenient with red for core limbs
    red_soft_bias = (profile == 'core_relaxed')
    
    # Slightly more permissive thresholds to reduce false 'unknown' reports
    red_threshold = 0.10 if red_soft_bias else 0.12
    yellow_threshold = 0.08
    green_threshold = 0.08

    # If analyzing a small mask (typical for arms/hands), be more permissive
    # so background bleed doesn't force an 'unknown' result.
    if mask is not None:
        masked_area = cv2.countNonZero(mask)
        if masked_area < 300:
            green_threshold = 0.06
            yellow_threshold = 0.06
            red_threshold = 0.10

    state = None
    # Soft red bias only for chest/torso profile
    if red_soft_bias and red_pct >= 0.07 and red_pct >= (yellow_pct * 0.75):
        state = 'red'
    elif red_pct > red_threshold:
        state = 'red'
    elif yellow_pct > yellow_threshold:
        state = 'yellow'
    elif green_pct > green_threshold:
        state = 'green'
    else:
        # Fallback: if one color has clear absolute dominance in pixel count
        # (useful for thin arm masks where background bleeds), prefer the
        # color with the largest absolute pixels provided it has at least
        # a couple pixels of evidence.
        max_count = max(red_pixels, yellow_pixels, green_pixels)
        if max_count >= max(2, min_colored_pixels):
            if max_count == red_pixels:
                state = 'red'
            elif max_count == yellow_pixels:
                state = 'yellow'
            else:
                state = 'green'
        else:
            # Last resort: pick the highest percentage
            max_pct = max(red_pct, yellow_pct, green_pct)
            if max_pct == red_pct:
                state = 'red'
            elif max_pct == yellow_pct:
                state = 'yellow'
            else:
                state = 'green'

    dominant_pct = max(red_pct, yellow_pct, green_pct)
    pixel_strength = min(1.0, total_colored / max(float(min_colored_pixels * 2), 1.0))
    confidence = float(np.clip((0.65 * dominant_pct) + (0.35 * pixel_strength), 0.0, 1.0))

    return {
        'state': state,
        'confidence': confidence,
        'coverage_ratio': coverage_ratio,
        'red_pct': red_pct,
        'yellow_pct': yellow_pct,
        'green_pct': green_pct,
        'total_colored': total_colored,
    }

def detect_color_state(roi_section, mask=None, min_colored_pixels=3, profile='strict'):
    """Backwards-compatible state-only wrapper."""
    return analyze_color_state(
        roi_section,
        mask=mask,
        min_colored_pixels=min_colored_pixels,
        profile=profile,
    )['state']

def get_limb_zone_bounds(limb_name, roi_w, roi_h):
    """Return integer zone bounds (x1, y1, x2, y2) for a limb in ROI coordinates."""
    if limb_name not in LIMB_ZONES:
        return None

    x1n, y1n, x2n, y2n = LIMB_ZONES[limb_name]
    x1 = max(0, min(roi_w - 1, int(x1n * roi_w)))
    x2 = max(1, min(roi_w, int(x2n * roi_w)))
    y1 = max(0, min(roi_h - 1, int(y1n * roi_h)))
    y2 = max(1, min(roi_h, int(y2n * roi_h)))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def get_limb_zone_analysis(roi_frame, color_mask, limb_name):
    """
    Fallback color detection in a fixed zone for a limb.
    Returns: 'green'/'yellow'/'red'/'unknown'
    """
    h, w = roi_frame.shape[:2]
    bounds = get_limb_zone_bounds(limb_name, w, h)
    if bounds is None:
        return {
            'state': 'unknown',
            'confidence': 0.0,
            'coverage_ratio': 0.0,
            'min_zone_pixels': 0,
        }
    x1, y1, x2, y2 = bounds

    zone_roi = roi_frame[y1:y2, x1:x2]
    zone_mask = color_mask[y1:y2, x1:x2]
    zone_area = zone_mask.shape[0] * zone_mask.shape[1]
    min_zone_pixels = max(3, int(zone_area * ZONE_MIN_COVERAGE_RATIO))
    profile = 'core_relaxed' if limb_name in CORE_LIMBS else 'strict'
    analysis = analyze_color_state(
        zone_roi,
        zone_mask,
        min_colored_pixels=min_zone_pixels,
        profile=profile,
    )
    analysis['min_zone_pixels'] = min_zone_pixels
    return analysis

def build_limb_template_masks(roi_w, roi_h):
    """
    Build traced limb masks so detection samples only model silhouette paths,
    reducing background bleed from nearby environment.
    """
    masks = {name: np.zeros((roi_h, roi_w), dtype=np.uint8) for name in ALL_LIMBS}

    def pt(xn, yn):
        # Apply global pixel shifts so templates can be quickly aligned
        return (int(xn * roi_w) + TEMPLATE_SHIFT_X, int(yn * roi_h) + TEMPLATE_SHIFT_Y)

    line_thick = max(2, int(min(roi_w, roi_h) * 0.05))
    leg_thick = max(2, int(min(roi_w, roi_h) * 0.045))
    arm_thick = max(2, int(min(roi_w, roi_h) * 0.045))
    # Apply extra leg thickness if configured
    leg_thick = leg_thick + int(max(0, LEG_THICK_ADJUST))

    # Head ring (apply per-limb offset)
    hx, hy = PER_LIMB_OFFSETS.get('head', (0, 0))
    head_center = pt(0.50, 0.10)
    head_center = (head_center[0] + hx, head_center[1] + hy)
    cv2.circle(masks['head'], head_center, max(3, int(min(roi_w, roi_h) * 0.10)), 255, thickness=line_thick)

    # Chest and torso are filled center regions (apply per-limb offsets)
    cx_off, cy_off = PER_LIMB_OFFSETS.get('chest', (0, 0))
    tx_off, ty_off = PER_LIMB_OFFSETS.get('torso', (0, 0))
    # Build chest polygon as floats so we can scale it about its centroid
    chest_poly = np.array([pt(0.36, 0.22), pt(0.64, 0.22), pt(0.66, 0.41), pt(0.34, 0.41)], dtype=np.float32)
    # Shrink chest polygon toward its centroid by CHEST_SHRINK_PX (approx)
    if CHEST_SHRINK_PX > 0:
        xs = chest_poly[:, 0]
        ys = chest_poly[:, 1]
        w = xs.max() - xs.min()
        h = ys.max() - ys.min()
        max_dim = max(w, h, 1.0)
        scale = max(0.0, 1.0 - (float(CHEST_SHRINK_PX) / float(max_dim)))
        centroid = chest_poly.mean(axis=0)
        chest_poly = (centroid + (chest_poly - centroid) * scale).astype(np.int32)
    else:
        chest_poly = chest_poly.astype(np.int32)
    chest_poly = chest_poly + np.array([cx_off, cy_off], dtype=np.int32)

    # Build torso polygon as floats so we can scale it about its centroid
    torso_poly = np.array([pt(0.38, 0.40), pt(0.62, 0.40), pt(0.60, 0.58), pt(0.40, 0.58)], dtype=np.float32)
    # Shrink torso polygon toward its centroid by TORSO_SHRINK_PX (approx)
    if TORSO_SHRINK_PX > 0:
        xs_t = torso_poly[:, 0]
        ys_t = torso_poly[:, 1]
        w_t = xs_t.max() - xs_t.min()
        h_t = ys_t.max() - ys_t.min()
        max_dim_t = max(w_t, h_t, 1.0)
        scale_t = max(0.0, 1.0 - (float(TORSO_SHRINK_PX) / float(max_dim_t)))
        centroid_t = torso_poly.mean(axis=0)
        torso_poly = (centroid_t + (torso_poly - centroid_t) * scale_t).astype(np.int32)
    else:
        torso_poly = torso_poly.astype(np.int32)
    torso_poly = torso_poly + np.array([tx_off, ty_off], dtype=np.int32)
    cv2.fillConvexPoly(masks['chest'], chest_poly, 255)
    cv2.fillConvexPoly(masks['torso'], torso_poly, 255)

    # Arms as traced curves (apply per-limb offsets)
    la_off = PER_LIMB_OFFSETS.get('left_arm', (0, 0))
    ra_off = PER_LIMB_OFFSETS.get('right_arm', (0, 0))
    left_arm_pts = np.array([pt(0.32, 0.23), pt(0.24, 0.34), pt(0.20, 0.48), pt(0.24, 0.60)], dtype=np.int32)
    right_arm_pts = np.array([pt(0.68, 0.23), pt(0.76, 0.34), pt(0.80, 0.48), pt(0.76, 0.60)], dtype=np.int32)
    left_arm_pts = left_arm_pts + np.array(la_off, dtype=np.int32)
    right_arm_pts = right_arm_pts + np.array(ra_off, dtype=np.int32)
    cv2.polylines(masks['left_arm'], [left_arm_pts], False, 255, thickness=arm_thick)
    cv2.polylines(masks['right_arm'], [right_arm_pts], False, 255, thickness=arm_thick)

    # Legs as traced lines (apply per-limb offsets)
    ll_off = PER_LIMB_OFFSETS.get('left_leg', (0, 0))
    rl_off = PER_LIMB_OFFSETS.get('right_leg', (0, 0))
    # Legs as traced lines (apply per-limb offsets). Grow the leg shape about
    # its centroid by LEG_GROW_PX to increase box/coverage.
    left_leg_pts = np.array([pt(0.45, 0.58), pt(0.38, 0.76), pt(0.32, 0.93)], dtype=np.float32)
    right_leg_pts = np.array([pt(0.55, 0.58), pt(0.62, 0.76), pt(0.68, 0.93)], dtype=np.float32)
    # Grow left leg
    if LEG_GROW_PX > 0:
        xs = left_leg_pts[:, 0]
        ys = left_leg_pts[:, 1]
        w = xs.max() - xs.min()
        h = ys.max() - ys.min()
        max_dim = max(w, h, 1.0)
        scale = 1.0 + (float(LEG_GROW_PX) / float(max_dim))
        centroid = left_leg_pts.mean(axis=0)
        left_leg_pts = (centroid + (left_leg_pts - centroid) * scale).astype(np.int32)
    else:
        left_leg_pts = left_leg_pts.astype(np.int32)

    # Grow right leg
    if LEG_GROW_PX > 0:
        xsr = right_leg_pts[:, 0]
        ysr = right_leg_pts[:, 1]
        wr = xsr.max() - xsr.min()
        hr = ysr.max() - ysr.min()
        max_dim_r = max(wr, hr, 1.0)
        scale_r = 1.0 + (float(LEG_GROW_PX) / float(max_dim_r))
        centroid_r = right_leg_pts.mean(axis=0)
        right_leg_pts = (centroid_r + (right_leg_pts - centroid_r) * scale_r).astype(np.int32)
    else:
        right_leg_pts = right_leg_pts.astype(np.int32)

    left_leg_pts = left_leg_pts + np.array(ll_off, dtype=np.int32)
    right_leg_pts = right_leg_pts + np.array(rl_off, dtype=np.int32)
    cv2.polylines(masks['left_leg'], [left_leg_pts], False, 255, thickness=leg_thick)
    cv2.polylines(masks['right_leg'], [right_leg_pts], False, 255, thickness=leg_thick)

    return masks

def initialize_arduino():
    # Initialize serial communication with Arduino
    global arduino_serial
    try:
        arduino_serial = serial.Serial(port='/dev/cu.usbmodem101', baudrate=9600, timeout=1)
        print(f"✓ Connected to Arduino on '/dev/cu.usbmodem101' at 9600 baud.")
        time.sleep(2)
    except Exception as e:
        print(f"⚠️  Failed to connect to Arduino: {e}")
        arduino_serial = None

def write_arduino(limb_type: LimbType, limb_status: LimbState):
    global arduino_serial

    if arduino_serial is None:
        print("⚠️  Arduino not connected")
        return

    try:
        limb = limb_type.value[0]  # '0', '1', '2', '3'
        status = limb_status.value[0]  # '0', '1', '2', '3', etc.
        arduino_serial.write(limb.encode())
        arduino_serial.write(status.encode())
        # time.sleep(0.75)  # Remove this - not needed anymore!
        print(f"Sent to Arduino: Limb={limb}, Status={status}")
        
    except Exception as e:
        print(f"⚠️  Failed to send data to Arduino: {e}")

def main():
    """
    Main function to monitor player model limb colors using contour detection
    """

    # Initialize Arduino connection at startup
    initialize_arduino()

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
    monitor_info = sct.monitors[1]  # Primary monitor
    monitor = {"top": 0, "left": 0, "width": monitor_info['width'], "height": monitor_info['height']}
    
    print(f"Screen resolution: {monitor['width']}x{monitor['height']}")
    print("Monitoring started...\n")
    
    # Track all expected limbs
    limbs_ever_seen = set()
    
    # Temporal smoothing - require seeing same state multiple times before changing
    detection_history = {}  # Store last N detections for each limb
    HISTORY_SIZE = 4  # Keep last 4 frames
    DAMAGE_CONFIDENCE = 2  # Need 2/4 frames for red/yellow (fast response to damage)
    SAFE_CONFIDENCE = 3  # Need 3/4 frames for green/missing (avoid false positives)
    consecutive_absent = {limb: 0 for limb in ALL_LIMBS}
    red_latch_remaining = {limb: 0 for limb in ALL_LIMBS}
    
    # Debug image flag
    debug_image_saved = False
    template_masks = build_limb_template_masks(ROI_WIDTH, ROI_HEIGHT)

    limb_object = LimbStatus();
    
    try:
        while True:
            # Capture screen
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Extract the region of interest
            roi_frame = frame[ROI_Y:ROI_Y+ROI_HEIGHT, ROI_X:ROI_X+ROI_WIDTH].copy()
            
            # Convert to HSV for better color detection
            hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
            
            # Create separate masks for each color to avoid merging
            # Use permissive hue range but require HIGH saturation to avoid
            # picking up environmental light interference (yellowing light, etc)
            
            # All colors - capture everything in the model
            # Require minimum saturation (80) to distinguish from washed-out environmental light
            color_mask = cv2.inRange(hsv_roi, np.array([0, 80, 38]), np.array([180, 255, 255]))
            
            # Debug: print pixel count for first frame
            if not debug_image_saved:
                c_cnt = cv2.countNonZero(color_mask)
                print(f"DEBUG: Total colored pixels: {c_cnt}")
            
            # Remove noise - very gentle to preserve limb shapes
            kernel = np.ones((2, 2), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            
            # Dilate slightly to connect nearby parts of same limb
            kernel_dilate = np.ones((3, 3), np.uint8)
            color_mask = cv2.dilate(color_mask, kernel_dilate, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            model_visible_ratio = cv2.countNonZero(color_mask) / float(ROI_WIDTH * ROI_HEIGHT)
            
            # Save debug images on first frame
            if SAVE_DEBUG_IMAGE and not debug_image_saved:
                cv2.imwrite('debug_captured_area.png', roi_frame)
                cv2.imwrite('debug_mask.png', color_mask)  # Save the mask too
                
                # Save contours visualization
                debug_contours = roi_frame.copy()
                cv2.drawContours(debug_contours, contours, -1, (0, 255, 255), 1)
                for i, contour in enumerate(contours):
                    if cv2.contourArea(contour) >= MIN_CONTOUR_AREA:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            limb_name = identify_limb(contour, ROI_WIDTH, ROI_HEIGHT)
                            cv2.putText(debug_contours, limb_name, (cx-10, cy), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                for limb_name in ALL_LIMBS:
                    contours_mask, _ = cv2.findContours(template_masks[limb_name], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(debug_contours, contours_mask, -1, (255, 200, 0), 1)
                cv2.imwrite('debug_contours.png', debug_contours)
                
                print(f"✓ Saved debug images: debug_captured_area.png, debug_mask.png, debug_contours.png")
                print(f"  Check these to see what is being detected\n")
                debug_image_saved = True
            
            # Primary traced-mask detection for all limbs
            current_states = {}
            frame_signal_conf = {limb: 0.0 for limb in ALL_LIMBS}
            for limb_name in ALL_LIMBS:
                limb_mask = template_masks[limb_name]
                mask_pixels = cv2.countNonZero(limb_mask)
                # Be permissive for small masks (arms/hands) while keeping
                # a sensible minimum for larger masks.
                min_pixels = max(2, int(mask_pixels * TEMPLATE_MASK_MIN_PIXELS_RATIO))
                profile = 'core_relaxed' if limb_name in CORE_LIMBS else 'strict'
                limb_analysis = analyze_color_state(
                    roi_frame,
                    mask=limb_mask,
                    min_colored_pixels=min_pixels,
                    profile=profile,
                )
                limb_state = limb_analysis['state']
                limb_conf = limb_analysis['confidence']

                if limb_state in ['green', 'yellow', 'red']:
                    current_states[limb_name] = limb_state
                    frame_signal_conf[limb_name] = limb_conf
                    limbs_ever_seen.add(limb_name)
                else:
                    # Print debug info for unknown/missing limbs to diagnose detection
                    print(f"  {limb_name}: {limb_state} (conf: {limb_conf:.2f}, colored: {limb_analysis['total_colored']}, "
                          f"green%: {limb_analysis['green_pct']:.3f}, yellow%: {limb_analysis['yellow_pct']:.3f}, "
                          f"red%: {limb_analysis['red_pct']:.3f}, coverage: {limb_analysis['coverage_ratio']:.3f})")
            
            # Update detection history for temporal smoothing
            for limb_name in ALL_LIMBS:
                if limb_name not in detection_history:
                    detection_history[limb_name] = []
                
                # Add current detection to history
                if limb_name in current_states:
                    detection_history[limb_name].append(current_states[limb_name])
                    consecutive_absent[limb_name] = 0
                else:
                    # If we've seen this limb before, use occlusion-aware missing handling.
                    if limb_name in limbs_ever_seen:
                        if model_visible_ratio < MIN_MODEL_VISIBLE_RATIO:
                            # Too little model is visible; likely occlusion/background interference.
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
                
                # Keep only last N frames
                if len(detection_history[limb_name]) > HISTORY_SIZE:
                    detection_history[limb_name].pop(0)
            
            # Determine stable state for each limb (majority vote from history)
            stable_states = {}
            stable_confidence = {}
            for limb_name in ALL_LIMBS:
                if limb_name not in detection_history or len(detection_history[limb_name]) == 0:
                    continue
                
                # Count occurrences of each state
                state_counts = Counter(detection_history[limb_name])
                
                # Get most common state
                most_common_state, count = state_counts.most_common(1)[0]
                
                # Use different thresholds based on state type.
                # Red/yellow (damage) = fast response, other states = more conservative.
                if most_common_state in ['red', 'yellow']:
                    required_confidence = DAMAGE_CONFIDENCE
                else:
                    required_confidence = SAFE_CONFIDENCE
                
                # Only use this state if it appears enough times
                if count >= required_confidence:
                    stable_states[limb_name] = most_common_state
                    temporal_conf = count / max(len(detection_history[limb_name]), 1)
                    stable_confidence[limb_name] = (0.5 * temporal_conf) + (0.5 * frame_signal_conf.get(limb_name, 0.0))

            # Immediate red override: if this frame has confident red, show it immediately.
            for limb_name in ALL_LIMBS:
                current_state = current_states.get(limb_name)
                current_conf = frame_signal_conf.get(limb_name, 0.0)
                if current_state == 'red' and current_conf >= RED_IMMEDIATE_CONF:
                    stable_states[limb_name] = 'red'
                    stable_confidence[limb_name] = max(stable_confidence.get(limb_name, 0.0), current_conf)
                    red_latch_remaining[limb_name] = RED_LATCH_FRAMES
                elif red_latch_remaining[limb_name] > 0:
                    # Keep red visible for a couple frames to avoid missing quick spikes.
                    stable_states[limb_name] = 'red'
                    stable_confidence[limb_name] = max(stable_confidence.get(limb_name, 0.0), 0.45)
                    red_latch_remaining[limb_name] -= 1
            
            # Print status of all limbs every frame
            print("\n--- Frame Update ---")
            
            # Always print all limbs with current stable state + confidence.
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
                if (limb_name == "left_arm" or limb_name == "left_leg" or limb_name == "right_arm" or limb_name == "right_leg"):
                    limb_type = LimbType[limb_name.upper()]
                    limb_status = LimbState[color_state.upper()]
                    limb_object.setLimbStatus(limb_type=limb_type, limb_state=limb_status)

                    if (limb_object.hasStatusChanged(limb_type=limb_type)):
                        write_arduino(limb_type, limb_status)
            
            # Show visualization window if enabled
            if SHOW_WINDOW:
                if SHOW_ONLY_ZOOM:
                    # Only show the zoomed view - easier to position
                    if roi_frame.size > 0:
                        # Add position info on the zoomed view
                        display_zoom = roi_frame.copy()
                        cv2.putText(display_zoom, f"Pos: ({ROI_X},{ROI_Y}) Size: {ROI_WIDTH}x{ROI_HEIGHT}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(display_zoom, "Adjust ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.imshow('What the detector sees - Should show player model', 
                                   cv2.resize(display_zoom, (600, 825)))
                else:
                    # Show both windows
                    # Draw detection box on full screen
                    frame_display = frame.copy()
                    cv2.rectangle(frame_display, (ROI_X, ROI_Y), 
                                 (ROI_X + ROI_WIDTH, ROI_Y + ROI_HEIGHT), (0, 255, 255), 4)
                    cv2.putText(frame_display, f"<-- MOVE THIS YELLOW BOX TO COVER PLAYER MODEL", 
                               (ROI_X + ROI_WIDTH + 10, ROI_Y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame_display, f"Position: ({ROI_X},{ROI_Y}) | Size: {ROI_WIDTH}x{ROI_HEIGHT}", 
                               (ROI_X, ROI_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Show reduced size screen view
                    scale = 0.5 if monitor['width'] > 2000 else 0.6
                    cv2.imshow('YOUR SCREEN - Find the player model and move yellow box to it', 
                               cv2.resize(frame_display, 
                               (int(monitor['width'] * scale), int(monitor['height'] * scale))))
                    
                    # Show zoomed ROI - bigger so you can actually see it
                    if roi_frame.size > 0:
                        cv2.imshow('Inside the Yellow Box (what will be analyzed)', 
                                   cv2.resize(roi_frame, (600, 825)))
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Small delay to reduce CPU usage and output spam
            time.sleep(0.5)  # Update twice per second
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        print("Goodbye!")
    finally:
        if SHOW_WINDOW:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
