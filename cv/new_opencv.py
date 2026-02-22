import cv2
import numpy as np
from mss import mss
import time
from collections import Counter
from enum import Enum
import serial
from new_find_model import load_limb_templates

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
ROI_X = 52
ROI_Y = 135
ROI_WIDTH = 115
ROI_HEIGHT = 269

# ROI adjustment controls
MOVE_STEP = 5       # Pixels to move per keypress
SIZE_STEP = 10      # Pixels to resize per keypress

# VISUALIZATION MODE
SHOW_WINDOW = False
SHOW_ONLY_ZOOM = False
SAVE_DEBUG_IMAGE = True
DEBUG_MODE = False
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

    print("=" * 60)
    print("Starting Player Health Monitor...")
    print("=" * 60)
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
            
            # Small delay
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        print("Goodbye!")

if __name__ == "__main__":
    main()