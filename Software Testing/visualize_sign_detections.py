#!/usr/bin/env python3
"""
Visual test script for street sign detection
Creates side-by-side comparisons of original images and detected masks
"""

import cv2
import numpy as np
import os
from PJGranieri import detect_street_signs

def create_visualization(image_path, output_dir="Output"):
    """
    Create a side-by-side visualization of original image and detected masks

    Args:
        image_path: Path to input image
        output_dir: Directory to save output images

    Returns:
        Path to saved visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Run detection
    print(f"\nProcessing: {os.path.basename(image_path)}")
    results = detect_street_signs(image_path, debug=False)

    # Load original image
    img_original = cv2.imread(image_path)
    if img_original is None:
        print(f"  Error: Could not load image")
        return None

    H, W = img_original.shape[:2]

    # Create annotated image (original with bounding boxes)
    img_annotated = img_original.copy()

    # Create combined mask visualization
    mask_combined = np.zeros((H, W, 3), dtype=np.uint8)

    # Process each detected sign
    detected_signs = []
    for street_name, is_present, street_sign_mask in results:
        if is_present and street_name:  # Only process actual detections
            detected_signs.append((street_name, street_sign_mask))

            # Convert mask to 2D for contour detection
            mask_2d = street_sign_mask.squeeze()

            # Find contours
            contours, _ = cv2.findContours(mask_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contours[0])

                # Draw green rectangle on annotated image
                cv2.rectangle(img_annotated, (x, y), (x+w, y+h), (0, 255, 0), 5)

                # Add text label
                label = street_name
                font_scale = min(w, h) / 200.0  # Scale font with sign size
                font_scale = max(0.5, min(font_scale, 2.0))  # Clamp between 0.5 and 2.0
                thickness = max(1, int(font_scale * 2))

                # Get text size for background
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                              font_scale, thickness)

                # Draw text background
                cv2.rectangle(img_annotated, (x, y - text_h - 20),
                            (x + text_w + 10, y - 5), (0, 255, 0), -1)

                # Draw text
                cv2.putText(img_annotated, label, (x + 5, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

                # Add mask to combined visualization (different color per sign)
                # Use different colors for different signs
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                         (255, 0, 255), (0, 255, 255)]
                color_idx = len(detected_signs) - 1
                color = colors[color_idx % len(colors)]

                mask_colored = np.zeros((H, W, 3), dtype=np.uint8)
                mask_colored[mask_2d > 0] = color
                mask_combined = cv2.addWeighted(mask_combined, 1.0, mask_colored, 0.7, 0)

                # Draw contour on mask visualization
                cv2.drawContours(mask_combined, contours, -1, color, 3)

                # Add label to mask
                cv2.rectangle(mask_combined, (x, y - text_h - 20),
                            (x + text_w + 10, y - 5), color, -1)
                cv2.putText(mask_combined, label, (x + 5, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    # Print detection summary
    if detected_signs:
        print(f"  ✅ Detected {len(detected_signs)} sign(s):")
        for name, _ in detected_signs:
            print(f"     - {name}")
    else:
        print(f"  ❌ No signs detected")
        # Add "NO SIGNS DETECTED" text to mask
        cv2.putText(mask_combined, "NO SIGNS DETECTED", (50, H//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # Resize images if too large (for display purposes)
    max_width = 1200
    if W > max_width:
        scale = max_width / W
        new_w = int(W * scale)
        new_h = int(H * scale)
        img_annotated = cv2.resize(img_annotated, (new_w, new_h))
        mask_combined = cv2.resize(mask_combined, (new_w, new_h))

    # Create side-by-side comparison
    comparison = np.hstack([img_annotated, mask_combined])

    # Add labels at the top
    label_height = 60
    labeled_comparison = np.zeros((comparison.shape[0] + label_height, comparison.shape[1], 3),
                                   dtype=np.uint8)
    labeled_comparison[label_height:, :] = comparison

    # Add text labels
    mid_point = comparison.shape[1] // 2
    cv2.putText(labeled_comparison, "ORIGINAL + DETECTIONS", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(labeled_comparison, "DETECTION MASKS", (mid_point + 20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Save comparison
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_comparison.jpg")
    cv2.imwrite(output_path, labeled_comparison)
    print(f"  💾 Saved: {output_path}")

    return output_path

def create_grid_visualization(image_paths, output_dir="Output"):
    """
    Create a grid visualization of multiple images

    Args:
        image_paths: List of paths to input images
        output_dir: Directory to save output
    """
    print("\n" + "="*80)
    print("CREATING GRID VISUALIZATION")
    print("="*80)

    visualizations = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            vis_path = create_visualization(img_path, output_dir)
            if vis_path:
                visualizations.append(cv2.imread(vis_path))

    if not visualizations:
        print("\nNo visualizations created")
        return

    # Create grid (2 columns)
    rows = []
    for i in range(0, len(visualizations), 2):
        if i + 1 < len(visualizations):
            # Resize to same height
            h1, w1 = visualizations[i].shape[:2]
            h2, w2 = visualizations[i+1].shape[:2]
            target_h = min(h1, h2)

            img1 = cv2.resize(visualizations[i], (int(w1 * target_h / h1), target_h))
            img2 = cv2.resize(visualizations[i+1], (int(w2 * target_h / h2), target_h))

            row = np.hstack([img1, img2])
        else:
            row = visualizations[i]
        rows.append(row)

    # Stack rows
    if rows:
        # Resize all rows to same width
        max_w = max(row.shape[1] for row in rows)
        resized_rows = []
        for row in rows:
            if row.shape[1] < max_w:
                # Pad with black
                padding = np.zeros((row.shape[0], max_w - row.shape[1], 3), dtype=np.uint8)
                row = np.hstack([row, padding])
            resized_rows.append(row)

        grid = np.vstack(resized_rows)

        # Save grid
        grid_path = os.path.join(output_dir, "all_detections_grid.jpg")
        cv2.imwrite(grid_path, grid)
        print(f"\n{'='*80}")
        print(f"✅ Grid visualization saved: {grid_path}")
        print(f"{'='*80}\n")

if __name__ == "__main__":
    # Test images - you can modify this list
    test_images = [
        "Data/IMG_4108.jpg",
        "Data/IMG_4110.jpg",
        "Data/IMG_1473.jpg",
        "Data/IMG_1480.JPG",
        "Data/IMG_1482.JPG",
        "Data/IMG_1483.JPG",
        "Data/upright_street_signs.png",
        "Data/IMG_1490.JPG",
    ]

    # Filter to only existing images
    existing_images = [img for img in test_images if os.path.exists(img)]

    if not existing_images:
        print("No test images found!")
        print("Available images:")
        if os.path.exists("Data"):
            for f in os.listdir("Data"):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    print(f"  Data/{f}")
    else:
        # Create individual visualizations
        for img_path in existing_images:
            create_visualization(img_path)

        # Create grid visualization
        create_grid_visualization(existing_images)

        print("\n✨ All visualizations complete!")
        print(f"Check the 'Output/' directory for results")
