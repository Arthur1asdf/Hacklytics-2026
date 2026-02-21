import cv2
import numpy as np
from mss import mss

print("=" * 60)
print("AUTO-DETECTING PLAYER MODEL...")
print("=" * 60)

# Capture full screen
sct = mss()
monitor_info = sct.monitors[1]
screenshot = sct.grab(monitor_info)
frame = np.array(screenshot)
frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

print(f"Screen resolution: {monitor_info['width']}x{monitor_info['height']}")

# Convert to HSV to find green colors
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Look for bright green (player model color)
green_lower = np.array([35, 40, 40])
green_upper = np.array([90, 255, 255])

# Create mask for green pixels
green_mask = cv2.inRange(hsv, green_lower, green_upper)

# Find contours of green areas
contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) == 0:
    print("\n❌ No green player model found on screen!")
    print("   Make sure the Tarkov video with green player model is visible")
    exit(1)

# Find the leftmost, topmost green region (player model in top-left)
best_x = float('inf')
best_y = float('inf')
best_contour = None

for contour in contours:
    area = cv2.contourArea(contour)
    if area < 50:  # Too small
        continue
    
    x, y, w, h = cv2.boundingRect(contour)
    
    # Player model should be relatively tall/thin
    aspect_ratio = h / w if w > 0 else 0
    
    # Looking for top-left green region
    if x < best_x or (x == best_x and y < best_y):
        best_x = x
        best_y = y
        best_contour = contour

if best_contour is None:
    print("\n❌ Could not identify player model")
    print("   Try making the video window larger")
    exit(1)

# Get bounding box with some padding
x, y, w, h = cv2.boundingRect(best_contour)

# Add padding to capture full model
padding_x = int(w * 0.3)
padding_y = int(h * 0.2)

roi_x = max(0, x - padding_x)
roi_y = max(0, y - padding_y)
roi_width = w + (2 * padding_x)
roi_height = h + (2 * padding_y)

print(f"\n✓ Found player model at position ({x}, {y})")
print(f"  Size: {w}x{h}")

# Save debug image showing what was found
debug_frame = frame.copy()
cv2.rectangle(debug_frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 255), 3)
cv2.putText(debug_frame, "Player Model Detected", (roi_x, roi_y - 10),
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
cv2.imwrite('detected_player_model.png', debug_frame)
print(f"  Saved: detected_player_model.png")

# Extract and save just the ROI
roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
cv2.imwrite('player_model_roi.png', roi)
print(f"  Saved: player_model_roi.png")

print("\n" + "=" * 60)
print("COPY THESE VALUES INTO cvopen.py:")
print("=" * 60)
print(f"ROI_X = {roi_x}")
print(f"ROI_Y = {roi_y}")
print(f"ROI_WIDTH = {roi_width}")
print(f"ROI_HEIGHT = {roi_height}")
print("=" * 60)
print("\nCheck 'player_model_roi.png' to verify it captured the full model!")
