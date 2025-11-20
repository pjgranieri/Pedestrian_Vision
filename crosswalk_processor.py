import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

# --- Helper function to draw detections ---
def _draw_detection(image, bbox, text):
    """
    Function to draw bounding boxes and text on an image.
    """
    (x, y, w, h) = bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 20)
    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# --- New functions for crosswalk detection ---

def detect_white_mask(image):
    """
    Detects bright white regions in the image, ignoring the sky.
    Returns a binary mask where white regions are white.
    """
    h, w, _ = image.shape
    # Create a mask to ignore the top 40% of the image (sky)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[int(h * 0.4):, :] = 255 # Keep only the bottom 60%
    # Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply the ROI mask to the grayscale image
    gray = cv2.bitwise_and(gray, gray, mask=roi_mask)
    # Use a high threshold to isolate only the brightest pixels
    _, mask = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
    # Use morphology to clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    return mask

# --- MODIFIED FUNCTION ---
def find_crosswalks(mask):
    """
    Analyzes the white mask to find groups of repeating horizontal bars.
    MODIFIED: Now returns the bounding box for the single largest horizontal bar (closest to viewer).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    horizontal_bars = []
    # --- 1. Filter contours to find individual bars ---
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        # Get bounding box info
        rect = cv2.minAreaRect(cnt)
        (w_rot, h_rot) = rect[1]

        if w_rot == 0 or h_rot == 0:
            continue

        width = max(w_rot, h_rot)
        height = min(w_rot, h_rot)
        aspect_ratio = width / height
        # Check if it's a long, thin horizontal bar
        if aspect_ratio > 3.0:
            extent = area / (width * height)
            if extent > 0.75:
                (x, y, w, h) = cv2.boundingRect(cnt)
                # Store the bounding box AND the area for comparison
                horizontal_bars.append((x, y, w, h, area))

    if not horizontal_bars:
        return []

    # --- 2. Find the single largest horizontal bar (closest line) ---
    # The largest bar by area is typically the closest due to perspective.
    largest_bar_with_area = max(horizontal_bars, key=lambda bar: bar[4])
    # Extract the bounding box (x, y, w, h)
    (x, y, w, h, _) = largest_bar_with_area
    # --- 3. Format the result with padding ---
    padding = 10
    x_min_pad = max(0, x - padding)
    y_min_pad = max(0, y - padding)
    x_max_pad = min(mask.shape[1], x + w + padding)
    y_max_pad = min(mask.shape[0], y + h + padding)
    w_group = x_max_pad - x_min_pad
    h_group = y_max_pad - y_min_pad
    # Return a list containing a single bounding box
    return [(x_min_pad, y_min_pad, w_group, h_group)]


# --- Main controller function ---

def detect_crosswalk(image_path, debug):
    """
    Main controller function to detect crosswalks.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read the image file at {image_path}")
            return ("Error", False, None)
    except Exception as e:
        print(f"An error occurred while reading the image: {e}")
        return ("Error", False, None)
    output_image = img.copy()
    final_result = False
    crosswalk_mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

    # 1. Get the white mask
    white_mask = detect_white_mask(img)
    # 2. Find the single closest crosswalk bar
    crosswalk_bboxes = find_crosswalks(white_mask)
    if crosswalk_bboxes:
        final_result = True
        # NOTE: The detection count will now always be 1 or 0
        print(f"Found the single largest crosswalk line.")
        for bbox in crosswalk_bboxes:
            # --- MODIFIED LABEL ---
            _draw_detection(output_image, bbox, "Closest Crosswalk Line")
            (x, y, w, h) = bbox
            cv2.rectangle(crosswalk_mask, (x, y), (x + w, y + h), (255), -1)
    else:
        print("No valid crosswalk line found.")
    # == visualization for debugging ==
    if debug:
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 3, 1)
        plt.imshow(white_mask, cmap='gray')
        # plt.title('White Mask')
        plt.axis('off')
        plt.tight_layout()

        plt.subplot(1, 3, 2)
        plt.imshow(crosswalk_mask, cmap='gray')
        # plt.title('Closest Line Mask')
        plt.axis('off')
        plt.tight_layout()

        output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 3, 3)
        plt.imshow(output_image_rgb)
        # plt.title('Closest Crosswalk Line Detection')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # cv2.imwrite("output_closest_crosswalk_line.jpg", output_image)
        # print(("Closest Crosswalk Line", final_result, crosswalk_mask))
    return ("Closest Crosswalk Line", final_result, crosswalk_mask)

# --- Example Execution ---
if __name__ == '__main__':
    # --- IMPORTANT: REPLACE THIS with the path to your image ---
    input_img = './Data/Stop Signal/IMG_1499.jpg'
    # input_img = 'image_28796b.jpg' # Use this if you uploaded the file
    print(f"--- Processing file: {input_img} ---")
    try:
        (name, is_present, mask) = detect_crosswalk(input_img, True)
        print(f"Detection: {name}, Present: {is_present}\n")
    except FileNotFoundError:
        print(f"\nError: Image not found at {input_img}")
        print("Please update the 'input_img' variable with a valid path.")
    except Exception as e:
        if "img is None" not in str(e):
            print(f"An error occurred: {e}")
