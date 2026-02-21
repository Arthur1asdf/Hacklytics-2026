import cv2
import numpy as np
from mss import mss
import time
from collections import Counter

# ============================================================
# ADJUST THESE VALUES FOR YOUR SCREEN/RESOLUTION  
# ============================================================
# Based on your screenshot at 2880x1864 resolution
# Player model is at top-left of YouTube video

ROI_X = 17       # Left edge where player model appears
ROI_Y = 105         # Below browser tabs, at video content
ROI_WIDTH = 50     # Width of player model
ROI_HEIGHT = 120    # Height of player model (head to feet)

# VISUALIZATION MODE
SHOW_WINDOW = False  # No GUI - runs in background
SHOW_ONLY_ZOOM = False
SAVE_DEBUG_IMAGE = True  # Saves one image to verify position
DEBUG_MODE = False  # Print all detections for debugging
# ============================================================

# Minimum contour area to avoid noise (adjust if getting too many/few detections)
MIN_CONTOUR_AREA = 15  # Very low to catch all limb sections

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
    
    # Define NON-OVERLAPPING color ranges - must match the thresholds used in main
    # VERY HIGH saturation/value to only detect bright player model, ignore environment
    
    # Green range (hue 45-85, VERY HIGH saturation/value)
    green_lower = np.array([45, 75, 95])
    green_upper = np.array([85, 255, 255])
    
    # Yellow range (hue 18-44, VERY HIGH saturation/value)
    yellow_lower = np.array([18, 75, 95])
    yellow_upper = np.array([44, 255, 255])
    
    # Orange range (hue 5-17, VERY HIGH saturation/value)
    orange_lower = np.array([5, 75, 95])
    orange_upper = np.array([17, 255, 255])
    
    # Pure red range (hue 0-5 and 170-180, VERY HIGH saturation/value)
    red_lower1 = np.array([0, 75, 95])
    red_upper1 = np.array([5, 255, 255])
    red_lower2 = np.array([170, 75, 95])
    red_upper2 = np.array([180, 255, 255])
    
    # Create masks
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Combine orange and red for "red" detection (but NOT yellow)
    red_combined = cv2.bitwise_or(orange_mask, red_mask)
    
    # Count pixels for each color - these are now non-overlapping
    green_pixels = cv2.countNonZero(green_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    red_pixels = cv2.countNonZero(red_combined)
    
    total_colored = green_pixels + yellow_pixels + red_pixels
    
    # Need minimum pixels to make a determination - very low threshold
    if total_colored < 3:
        return 'unknown'
    
    # Determine dominant color with clear priority
    # Check each color independently with very low thresholds (10%)
    total = max(total_colored, 1)
    
    red_pct = red_pixels / total
    yellow_pct = yellow_pixels / total
    green_pct = green_pixels / total
    
    # Return the dominant color (must be >10% to be considered)
    if red_pct > 0.10:
        return 'red'
    elif yellow_pct > 0.10:
        return 'yellow'
    elif green_pct > 0.10:
        return 'green'
    else:
        # If no color is dominant enough, pick the highest
        max_pct = max(red_pct, yellow_pct, green_pct)
        if max_pct == red_pct:
            return 'red'
        elif max_pct == yellow_pct:
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
    
    # Track all expected limbs
    ALL_LIMBS = ['head', 'chest', 'torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
    limbs_ever_seen = set()
    
    # Temporal smoothing - require seeing same state multiple times before changing
    detection_history = {}  # Store last N detections for each limb
    HISTORY_SIZE = 4  # Keep last 4 frames
    DAMAGE_CONFIDENCE = 2  # Need 2/4 frames for red/yellow (fast response to damage)
    SAFE_CONFIDENCE = 3  # Need 3/4 frames for green/missing (avoid false positives)
    
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
            
            # Convert to HSV for better color detection
            hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
            
            # Create separate masks for each color to avoid merging
            # Use VERY HIGH saturation/value to ONLY detect bright player model colors
            # This ignores shadows, background, and environmental interference
            
            # Green player parts (ONLY bright saturated green)
            green_mask = cv2.inRange(hsv_roi, np.array([45, 80, 100]), np.array([85, 255, 255]))
            
            # Yellow player parts (ONLY bright saturated yellow)
            yellow_mask = cv2.inRange(hsv_roi, np.array([18, 80, 100]), np.array([44, 255, 255]))
            
            # Orange/Red player parts (ONLY bright saturated red/orange)
            orange_mask = cv2.inRange(hsv_roi, np.array([5, 80, 100]), np.array([18, 255, 255]))
            red_mask1 = cv2.inRange(hsv_roi, np.array([0, 80, 100]), np.array([5, 255, 255]))
            red_mask2 = cv2.inRange(hsv_roi, np.array([170, 80, 100]), np.array([180, 255, 255]))
            red_mask = cv2.bitwise_or(cv2.bitwise_or(orange_mask, red_mask1), red_mask2)
            
            # Combine all player model colors
            color_mask = cv2.bitwise_or(cv2.bitwise_or(green_mask, yellow_mask), red_mask)
            
            # Remove noise - very gentle to preserve limb shapes
            kernel = np.ones((2, 2), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            
            # Dilate slightly to connect nearby parts of same limb
            kernel_dilate = np.ones((3, 3), np.uint8)
            color_mask = cv2.dilate(color_mask, kernel_dilate, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
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
                cv2.imwrite('debug_contours.png', debug_contours)
                
                print(f"✓ Saved debug images: debug_captured_area.png, debug_mask.png, debug_contours.png")
                print(f"  Check these to see what is being detected\n")
                debug_image_saved = True
            
            # Check each significant contour
            current_states = {}
            if DEBUG_MODE and len(contours) > 0:
                print(f"\nFound {len(contours)} contours in ROI (ROI size: {ROI_WIDTH}x{ROI_HEIGHT})")
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area < MIN_CONTOUR_AREA:
                    continue
                
                # Identify which limb this contour represents
                limb_name = identify_limb(contour, ROI_WIDTH, ROI_HEIGHT)
                
                if limb_name == "unknown":
                    continue
                
                # Track that we've seen this limb at least once
                limbs_ever_seen.add(limb_name)
                
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
                    M = cv2.moments(contour)
                    cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
                    cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
                    cx_norm = cx / ROI_WIDTH
                    cy_norm = cy / ROI_HEIGHT
                    print(f"  {limb_name}: {color_state} (area: {int(area)}, pos: {cx},{cy}, norm: {cx_norm:.2f},{cy_norm:.2f})")
                
                # Store current state (if limb appears multiple times, keep first detection)
                if limb_name not in current_states:
                    current_states[limb_name] = color_state
            
            # Update detection history for temporal smoothing
            for limb_name in ALL_LIMBS:
                if limb_name not in detection_history:
                    detection_history[limb_name] = []
                
                # Add current detection to history
                if limb_name in current_states:
                    detection_history[limb_name].append(current_states[limb_name])
                else:
                    # If we've seen this limb before, mark as 'missing'
                    if limb_name in limbs_ever_seen:
                        detection_history[limb_name].append('missing')
                    else:
                        detection_history[limb_name].append('unknown')
                
                # Keep only last N frames
                if len(detection_history[limb_name]) > HISTORY_SIZE:
                    detection_history[limb_name].pop(0)
            
            # Determine stable state for each limb (majority vote from history)
            stable_states = {}
            for limb_name in ALL_LIMBS:
                if limb_name not in detection_history or len(detection_history[limb_name]) == 0:
                    continue
                
                # Count occurrences of each state
                state_counts = Counter(detection_history[limb_name])
                
                # Get most common state
                most_common_state, count = state_counts.most_common(1)[0]
                
                # Use different thresholds based on state type
                # Red/yellow (damage) = fast response, green/missing = more conservative
                if most_common_state in ['red', 'yellow']:
                    required_confidence = DAMAGE_CONFIDENCE
                else:
                    required_confidence = SAFE_CONFIDENCE
                
                # Only use this state if it appears enough times
                if count >= required_confidence:
                    # Don't report 'unknown' states
                    if most_common_state != 'unknown':
                        stable_states[limb_name] = most_common_state
            
            # Print status of all limbs every frame
            print("\n--- Frame Update ---")
            
            # Check for missing limbs (ones we've seen before but aren't visible now)
            for limb_name in limbs_ever_seen:
                if limb_name in stable_states and stable_states[limb_name] == 'missing':
                    print(f"❌  {limb_name.replace('_', ' ').title()}: MISSING")
            
            # Print all currently detected limbs with stable states
            for limb_name in ALL_LIMBS:
                if limb_name in stable_states:
                    color_state = stable_states[limb_name]
                    if color_state == 'missing':
                        continue  # Already printed above
                    elif color_state == 'red':
                        print(f"🔴  {limb_name.replace('_', ' ').title()}: RED")
                    elif color_state == 'yellow':
                        print(f"🟡  {limb_name.replace('_', ' ').title()}: YELLOW")
                    elif color_state == 'green':
                        print(f"🟢  {limb_name.replace('_', ' ').title()}: GREEN")
            
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
