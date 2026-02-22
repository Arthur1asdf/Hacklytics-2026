import cv2
import numpy as np
import pickle

"""
STEP 1: Generate limb templates from a reference image
Run this once to create templates, then use them in your main script
"""

def create_limb_templates_from_image(image_path, roi_width, roi_height):
    """
    Extract limb shapes from a reference image using edge detection
    
    Args:
        image_path: Path to reference image of player model
        roi_width: Target width to scale templates to
        roi_height: Target height to scale templates to
    
    Returns:
        dict of limb_name -> binary mask
    """
    print("=" * 60)
    print("GENERATING LIMB TEMPLATES FROM REFERENCE IMAGE")
    print("=" * 60)
    
    # Read reference image
    ref_image = cv2.imread(image_path)
    if ref_image is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    print(f"✓ Loaded reference image: {ref_image.shape[1]}x{ref_image.shape[0]}")
    
    # Resize to target dimensions
    ref_image = cv2.resize(ref_image, (roi_width, roi_height))
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(ref_image, cv2.COLOR_BGR2HSV)
    
    # Create mask for ALL colored pixels (player model)
    # Lower saturation requirement to catch more of the model
    color_mask = cv2.inRange(hsv, np.array([0, 40, 30]), np.array([180, 255, 255]))
    
    # Clean up noise (reduce kernel size for thinner result)
    kernel = np.ones((2, 2), np.uint8)  # Smaller kernel = thinner
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Option 1: Use filled silhouette (recommended)
    # Just use the color mask directly - it's already filled!
    filled_mask = color_mask.copy()
    
    # Make it thinner by eroding (shrinking) the mask
    kernel_erode = np.ones((3, 3), np.uint8)
    filled_mask = cv2.erode(filled_mask, kernel_erode, iterations=2)  # Just a bit thinner
    
    # Optional: Dilate slightly to ensure coverage
    # Uncomment these lines if you want thicker masks:
    # kernel_dilate = np.ones((3, 3), np.uint8)
    # filled_mask = cv2.dilate(filled_mask, kernel_dilate, iterations=1)
    
    # Save full body mask
    cv2.imwrite('template_full_body_mask.png', filled_mask)
    print(f"✓ Saved: template_full_body_mask.png")
    
    # Now segment into individual limbs based on position
    height, width = filled_mask.shape
    limb_masks = {}
    
    # Define regions for each limb (normalized coordinates)
    limb_regions = {
        'head': (0.30, 0.00, 0.70, 0.13),
        'chest': (0.25, 0.13, 0.75, 0.34),
        'torso': (0.28, 0.34, 0.72, 0.44),
        'left_arm': (0.00, 0.15, 0.28, 0.55),
        'right_arm': (0.72, 0.15, 1.00, 0.55),
        'left_leg': (0.20, 0.43, 0.50, 1.00),
        'right_leg': (0.50, 0.43, 0.80, 1.00),
    }
    
    print("\n" + "=" * 60)
    print("LIMB REGION COORDINATES (Normalized)")
    print("=" * 60)
    for limb_name, (x1_norm, y1_norm, x2_norm, y2_norm) in limb_regions.items():
        print(f"'{limb_name}': ({x1_norm:.2f}, {y1_norm:.2f}, {x2_norm:.2f}, {y2_norm:.2f})")
    
    print("\n" + "=" * 60)
    print("LIMB REGION COORDINATES (Pixels)")
    print("=" * 60)
    
    for limb_name, (x1_norm, y1_norm, x2_norm, y2_norm) in limb_regions.items():
        # Convert normalized coords to pixels
        x1 = int(x1_norm * width)
        x2 = int(x2_norm * width)
        y1 = int(y1_norm * height)
        y2 = int(y2_norm * height)
        
        print(f"{limb_name:12} -> x1={x1:4}, y1={y1:4}, x2={x2:4}, y2={y2:4} (width={x2-x1}, height={y2-y1})")
        
        # Extract limb region from full filled mask
        limb_mask = np.zeros((height, width), dtype=np.uint8)
        limb_mask[y1:y2, x1:x2] = filled_mask[y1:y2, x1:x2]
        
        # Count pixels in this limb region
        pixel_count = cv2.countNonZero(limb_mask)
        
        # Store mask
        limb_masks[limb_name] = limb_mask
        
        # Save individual limb masks for visualization
        cv2.imwrite(f'template_{limb_name}_mask.png', limb_mask)
        print(f"  ✓ Generated template: {pixel_count} pixels")
    
    # Create composite visualization
    composite = np.zeros((height, width, 3), dtype=np.uint8)
    colors = {
        'head': (255, 0, 0),      # Blue
        'chest': (0, 255, 255),   # Yellow
        'torso': (0, 165, 255),   # Orange
        'left_arm': (0, 255, 0),  # Green
        'right_arm': (255, 0, 255), # Magenta
        'left_leg': (255, 255, 0), # Cyan
        'right_leg': (128, 0, 128), # Purple
    }
    
    for limb_name, mask in limb_masks.items():
        color = colors.get(limb_name, (255, 255, 255))
        composite[mask > 0] = color
    
    # Overlay on original image
    overlay = cv2.addWeighted(ref_image, 0.6, composite, 0.4, 0)
    cv2.imwrite('template_limbs_overlay.png', overlay)
    print(f"\n✓ Saved: template_limbs_overlay.png")
    
    # Save templates to file
    with open('limb_templates.pkl', 'wb') as f:
        pickle.dump({
            'masks': limb_masks,
            'width': roi_width,
            'height': roi_height
        }, f)
    print(f"✓ Saved: limb_templates.pkl")
    
    print("\n" + "=" * 60)
    print("TEMPLATE GENERATION COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - limb_templates.pkl (use this in your main script)")
    print("  - template_limbs_overlay.png (visual verification)")
    print("  - template_*_mask.png (individual limb masks)")
    
    print("\n" + "=" * 60)
    print("COPY THESE VALUES TO cvopen.py:")
    print("=" * 60)
    print("\nPaste this into your cvopen.py limb_regions dictionary:")
    print("\nlimb_regions = {")
    for limb_name, coords in limb_regions.items():
        print(f"    '{limb_name}': {coords},")
    print("}")
    
    print("\nNext step: Update cvopen.py to use these templates!")
    
    return limb_masks


def load_limb_templates():
    """
    Load pre-generated limb templates from file
    
    Returns:
        dict of limb_name -> binary mask
    """
    try:
        with open('limb_templates.pkl', 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Loaded limb templates: {data['width']}x{data['height']}")
        return data['masks']
    except FileNotFoundError:
        print("❌ limb_templates.pkl not found!")
        print("   Run create_limb_templates_from_image() first")
        return None


# USAGE EXAMPLE
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LIMB TEMPLATE GENERATOR")
    print("=" * 60)
    print("\nThis script creates accurate limb templates from a reference image.")
    print("\nSTEPS:")
    print("1. Take a screenshot of the player model (GREEN health state)")
    print("2. Crop it to just the player model area (use ROI coordinates)")
    print("3. Save as 'player_reference.png'")
    print("4. Run this script")
    print("\n" + "=" * 60)
    
    # Check if reference image exists
    import os
    if not os.path.exists('player_reference.png'):
        print("\n❌ 'player_reference.png' not found!")
        print("\nCreate it by:")
        print("  1. Run your detection script to get ROI coordinates")
        print("  2. Take a screenshot of the game")
        print("  3. Crop to ROI area and save as 'player_reference.png'")
        print("  4. Re-run this script")
    else:
        # Set these to match your ROI dimensions from cvopen.py
        TARGET_WIDTH = 150
        TARGET_HEIGHT = 365
        masks = create_limb_templates_from_image(
            'player_reference.png',
            TARGET_WIDTH,
            TARGET_HEIGHT
        )
        
        if masks:
            print("\n✅ SUCCESS! Templates generated.")
            print("\nNow update cvopen.py to use these templates:")
            print("  1. Add: from new_find_model import load_limb_templates")
            print("  2. Replace: template_masks = build_limb_template_masks(...)")
            print("     With:    template_masks = load_limb_templates()")