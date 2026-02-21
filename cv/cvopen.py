import cv2
import numpy as np
from mss import mss
import time

# ============================================================
# ADJUST THESE VALUES FOR YOUR SCREEN/RESOLUTION  
# ============================================================
# Based on your screenshot at 2880x1864 resolution
# Player model is at top-left of YouTube video

ROI_X = 30       # Left edge where player model appears
ROI_Y = 220         # Below browser tabs, at video content
ROI_WIDTH = 150     # Width of player model
ROI_HEIGHT = 215    # Height of player model (head to feet)

# VISUALIZATION MODE
SHOW_WINDOW = False  # No GUI - runs in background
SHOW_ONLY_ZOOM = False
SAVE_DEBUG_IMAGE = True  # Saves one image to verify position
DEBUG_MODE = True  # Print all detections for debugging
# ============================================================

# Minimum contour area to avoid noise (adjust if getting too many/few detections)
MIN_CONTOUR_AREA = 50  # Lowered to catch smaller limb sections

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
    
    # Identify based on position
    # Head: top center, roughly circular
    if cy_norm < 0.2 and 0.3 < cx_norm < 0.7:
        return "head"
    
    # Arms: middle height, left or right side
    elif 0.2 < cy_norm < 0.5:
        if cx_norm < 0.35:
            return "left_arm"
        elif cx_norm > 0.65:
            return "right_arm"
        else:
            return "torso"
    
    # Torso: center, middle section
    elif 0.3 < cy_norm < 0.6 and 0.35 < cx_norm < 0.65:
        return "torso"
    
    # Legs: bottom half
    elif cy_norm > 0.55:
        if cx_norm < 0.5:
            return "left_leg"
        else:
            return "right_leg"
    
    return "unknown"

def detect_color_state(roi_section, mask=None):
    """
    Detect if the limb is green (healthy), yellow (warning), or red (critical)
    Returns: 'green', 'yellow', 'red', or 'unknown'
    """
    if roi_section.size == 0:
        return 'unknown'
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(roi_section, cv2.COLOR_BGR2HSV)
    
    # If mask is provided, only analyze masked pixels
    if mask is not None:
        hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
    
    # Define VERY lenient color ranges in HSV for better detection
    # These ranges overlap intentionally to catch color variations
    
    # Green range - very wide (hue 40-85 covers most greens)
    green_lower = np.array([40, 30, 30])  # Lower saturation/value threshold
    green_upper = np.array([85, 255, 255])
    
    # Yellow-green range (catches yellow and yellow-green)
    yellow_lower = np.array([20, 30, 50])  # Much lower thresholds
    yellow_upper = np.array([45, 255, 255])  # Overlaps with green
    
    # Orange-red range (catches orange to red)
    orange_lower = np.array([5, 30, 50])
    orange_upper = np.array([25, 255, 255])
    
    # Pure red range (wraps around hue 0/180)
    red_lower1 = np.array([0, 30, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 30, 50])
    red_upper2 = np.array([180, 255, 255])
    
    # Create masks
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Combine orange and red for "red" detection
    red_combined = cv2.bitwise_or(orange_mask, red_mask)
    
    # Count pixels for each color
    green_pixels = cv2.countNonZero(green_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    red_pixels = cv2.countNonZero(red_combined)
    
    total_colored = green_pixels + yellow_pixels + red_pixels
    
    # Need minimum pixels to make a determination
    if total_colored < 15:
        return 'unknown'
    
    # Determine dominant color - prioritize damage alerts (red/yellow)
    # Use percentage-based detection instead of absolute counts
    total = max(total_colored, 1)  # Avoid division by zero
    
    red_pct = red_pixels / total
    yellow_pct = yellow_pixels / total
    green_pct = green_pixels / total
    
    # Prioritize red/yellow detection (more sensitive to damage)
    if red_pct > 0.25:  # If 25%+ pixels are red/orange
        return 'red'
    elif yellow_pct > 0.30:  # If 30%+ pixels are yellow
        return 'yellow'
    elif green_pct > 0.35:  # If 35%+ pixels are green
        return 'green'
    else:
        # Mixed or unclear - report most dominant
        if red_pixels > max(yellow_pixels, green_pixels):
            return 'red'
        elif yellow_pixels > green_pixels:
            return 'yellow'
        else:
            return 'green'

def main():
    """
    Main function to monitor player model limb colors using contour detection
    """
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
    
    # Store previous states to avoid spam
    previous_states = {}
    
    # Stabilize detections - only report after consecutive frames
    detection_buffer = {}
    REQUIRED_CONSECUTIVE_FRAMES = 3
    
    # Debug image flag
    debug_image_saved = False
    
    try:
        while True:
            # Capture screen
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Extract the region of interest
            roi_frame = frame[ROI_Y:ROI_Y+ROI_HEIGHT, ROI_X:ROI_X+ROI_WIDTH].copy()
            
            # Save debug image on first frame
            if SAVE_DEBUG_IMAGE and not debug_image_saved:
                cv2.imwrite('debug_captured_area.png', roi_frame)
                print(f"✓ Saved debug image: debug_captured_area.png")
                print(f"  Check this image to see if player model is captured\n")
                debug_image_saved = True
            
            # Convert to HSV for better color detection
            hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
            
            # Create a mask for all colored pixels (green, yellow, red)
            # This will find the player model outline
            lower_bound = np.array([0, 50, 50])
            upper_bound = np.array([180, 255, 255])
            color_mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)
            
            # Remove noise
            kernel = np.ones((3, 3), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check each significant contour
            current_states = {}
            if DEBUG_MODE and len(contours) > 0:
                print(f"Found {len(contours)} contours in ROI")
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area < MIN_CONTOUR_AREA:
                    continue
                
                # Identify which limb this contour represents
                limb_name = identify_limb(contour, ROI_WIDTH, ROI_HEIGHT)
                
                if limb_name == "unknown":
                    continue
                
                # Create a mask for this specific contour
                limb_mask = np.zeros(roi_frame.shape[:2], dtype=np.uint8)
                cv2.drawContours(limb_mask, [contour], -1, 255, -1)
                
                # Get bounding box for this limb
                x, y, w, h = cv2.boundingRect(contour)
                limb_roi = roi_frame[y:y+h, x:x+w]
                limb_mask_section = limb_mask[y:y+h, x:x+w]
                
                # Detect color state for this limb
                color_state = detect_color_state(limb_roi, limb_mask_section)
                
                if DEBUG_MODE:
                    print(f"  {limb_name}: {color_state} (area: {int(area)})")
                
                # Store current state
                if limb_name not in current_states:
                    current_states[limb_name] = color_state
            
            # Update detection buffer and print status changes
            for limb_name, color_state in current_states.items():
                if color_state in ['yellow', 'red']:
                    # Initialize buffer for this limb if needed
                    if limb_name not in detection_buffer:
                        detection_buffer[limb_name] = {'state': color_state, 'count': 1}
                    elif detection_buffer[limb_name]['state'] == color_state:
                        detection_buffer[limb_name]['count'] += 1
                    else:
                        # State changed, reset counter
                        detection_buffer[limb_name] = {'state': color_state, 'count': 1}
                    
                    # Only print if we've seen this state consistently
                    if detection_buffer[limb_name]['count'] >= REQUIRED_CONSECUTIVE_FRAMES:
                        if limb_name not in previous_states or previous_states[limb_name] != color_state:
                            print(f"⚠️  {limb_name.replace('_', ' ').title()}: {color_state.upper()}")
                            previous_states[limb_name] = color_state
                else:
                    # Limb is green or unknown - clear from buffer and previous states
                    if limb_name in detection_buffer:
                        del detection_buffer[limb_name]
                    if limb_name in previous_states:
                        print(f"✓  {limb_name.replace('_', ' ').title()}: RECOVERED")
                        del previous_states[limb_name]
            
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
            
            # Small delay to reduce CPU usage
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        print("Goodbye!")
    finally:
        if SHOW_WINDOW:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
