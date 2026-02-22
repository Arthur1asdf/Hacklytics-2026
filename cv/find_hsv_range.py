"""
HSV Range Finder - Interactive tool to discover player model color ranges
Adjusts trackbars to find the right HSV ranges, shows matching pixels in real-time
"""
import cv2
import numpy as np
from mss import mss

# Capture settings (must match cvopen.py)
ROI_X = 17
ROI_Y = 35
ROI_WIDTH = 200
ROI_HEIGHT = 370

def nothing(x):
    pass

# Create window with trackbars
cv2.namedWindow('HSV Range Finder')

# Create trackbars for lower HSV bounds
cv2.createTrackbar('LH', 'HSV Range Finder', 0, 180, nothing)
cv2.createTrackbar('LS', 'HSV Range Finder', 0, 255, nothing)
cv2.createTrackbar('LV', 'HSV Range Finder', 0, 255, nothing)

# Create trackbars for upper HSV bounds
cv2.createTrackbar('UH', 'HSV Range Finder', 180, 180, nothing)
cv2.createTrackbar('US', 'HSV Range Finder', 255, 255, nothing)
cv2.createTrackbar('UV', 'HSV Range Finder', 255, 255, nothing)

sct = mss()
monitor_info = sct.monitors[1]
monitor = {"top": 0, "left": 0, "width": monitor_info['width'], "height": monitor_info['height']}

print("HSV Range Finder started!")
print("Use trackbars to find the color ranges")
print("Press 'q' to quit")
print("\nMove the trackbars until you see green pixels lighting up in the window")
print("When you find the right range, note the LH, LS, LV, UH, US, UV values\n")

while True:
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    # Extract ROI
    roi_frame = frame[ROI_Y:ROI_Y+ROI_HEIGHT, ROI_X:ROI_X+ROI_WIDTH].copy()
    hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    
    # Get trackbar values
    lh = cv2.getTrackbarPos('LH', 'HSV Range Finder')
    ls = cv2.getTrackbarPos('LS', 'HSV Range Finder')
    lv = cv2.getTrackbarPos('LV', 'HSV Range Finder')
    uh = cv2.getTrackbarPos('UH', 'HSV Range Finder')
    us = cv2.getTrackbarPos('US', 'HSV Range Finder')
    uv = cv2.getTrackbarPos('UV', 'HSV Range Finder')
    
    # Create mask with current ranges
    lower = np.array([lh, ls, lv])
    upper = np.array([uh, us, uv])
    mask = cv2.inRange(hsv_roi, lower, upper)
    
    # Show result
    result = cv2.bitwise_and(roi_frame, roi_frame, mask=mask)
    
    # Count pixels
    pixel_count = cv2.countNonZero(mask)
    
    # Draw text on image
    cv2.putText(result, f"Pixels found: {pixel_count}", (5, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(result, f"Range: H({lh}-{uh}) S({ls}-{us}) V({lv}-{uv})", (5, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    # Display
    cv2.imshow('HSV Range Finder', result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"\nFinal ranges found:")
        print(f"Lower: [{lh}, {ls}, {lv}]")
        print(f"Upper: [{uh}, {us}, {uv}]")
        print(f"Total pixels: {pixel_count}")
        break

cv2.destroyAllWindows()
