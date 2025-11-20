import cv2 
import numpy as np
import easyocr 
import matplotlib.pyplot as plt
import glob
import torch

try:
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())  # initialize EasyOCR reader | no GPU
except Exception as e:
    exit()

# function to detect red regions in the image
def detect_red_mask(image):
    """
    Detects all the red in the image. Returns a binary mask where red regions are white.
    """
    # convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)

    # lower red range (0-15)
    lower_red1 = np.array([0, 30, 30])   
    upper_red1 = np.array([15, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    # upper red range (240-360)
    lower_red2 = np.array([240, 30, 30])
    upper_red2 = np.array([255, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((3,3), np.uint8)

    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=5)
    

    return red_mask

def is_octagon(mask):
    """
    Check the red mask for contours with roughly 8 vertices.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    octagon_boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area < 100: 
            continue

        (x, y, w, h) = cv2.boundingRect(cnt)
        
        # check circularity to filter out circles
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        if circularity > 0.95:
            continue

        # check extent to filter out squares/rectangles
        if w == 0 or h == 0:
            continue 
        extent = area / (w * h)
        # perfect square = 1.0
        # perfect octagon = ~0.83
        if extent > 0.85:
            continue

        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.03 * perimeter 
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # check for vertices count between 6 and 9
        if len(approx) >= 6 and len(approx) <= 9:
            (x_approx, y_approx, w_approx, h_approx) = cv2.boundingRect(approx)
            octagon_boxes.append((x_approx, y_approx, w_approx, h_approx))

    return octagon_boxes

def has_stop_text(image):
    """
    Define ROI and check for "STOP".
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray, detail=1)

    # returns a list of (bbox, text, confidence) tuples 
    for (bbox_points, text, confidence) in results:
        # check if the detected text contains "STOP"
        if "STOP" in text.upper():
            # [top_left, top_right, bottom_right, bottom_left]
            
            top_left = bbox_points[0]
            bottom_right = bbox_points[2]
            
            # get the top-left corner
            x = int(top_left[0])
            y = int(top_left[1])

            # get the width and height
            w = int(bottom_right[0] - top_left[0])
            h = int(bottom_right[1] - top_left[1])
            
            return (True, (x, y, w, h)) 
    return (False, None)

# function to draw bounding boxes and text
def _draw_detection(image, bbox, text):
    (x, y, w, h) = bbox
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 10)
    cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# function to determine if a stop sign is present
def stop_sign_detected(image_path, debug):
    """
    Main controller function.
    """
    # load image
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read the image file at {image_path}")
            return False
    except Exception as e:
        print(f"An error occurred while reading the image: {e}")
        return False
    
    output_image = img.copy()
    final_result = False

    stop_sign_mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

    # get the single red mask
    red_mask = detect_red_mask(img)
    # get the list of all potential octagons
    potential_signs = is_octagon(red_mask) 
    # loop through all candidates and check for "STOP"
    for shape_bbox in potential_signs:
        (x, y, w, h) = shape_bbox
        # create ROI with padding
        padding = 5
        roi = img[max(0, y - padding):min(img.shape[0], y + h + padding), 
                  max(0, x - padding):min(img.shape[1], x + w + padding)]
        if roi.size == 0:
            continue # skip if ROI empty

        # run EasyOCR on the cropped ROI
        results = reader.readtext(roi, detail=1)
        
        found_text_in_roi = False
        for (bbox_points, text, confidence) in results:
            clean_text = text.upper().strip()
            print(f"CR found: '{clean_text}' (Confidence: {confidence:.2f})")

            if clean_text == "STOP" and confidence > 0.6: # 60% threshold
                found_text_in_roi = True
                print(f"Found 'STOP' at {(x, y, w, h)} with confidence {confidence:.2f}\n")
                break

        # print(f"Text Check on candidate at {(x,y, w, h)}: {found_text_in_roi}")

        if found_text_in_roi:
            final_result = True
            _draw_detection(output_image, shape_bbox, " ")
            cv2.rectangle(stop_sign_mask, (x, y), (x+w, y+h), (255), -1)
            break 
    
    if not final_result:
        print("No valid stop sign found.")

    if debug:
        # == visualization for debugging ==
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 3, 1)
        plt.imshow(red_mask, cmap='gray')
        plt.title('Red Mask')
        plt.axis('off')
        plt.subplot(1, 3, 2) # (rows, cols, index)
        plt.imshow(stop_sign_mask, cmap='gray')
        plt.title('Stop Sign Mask')
        plt.axis('off')
        # Convert BGR to RGB 
        output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 3, 3)
        plt.imshow(output_image_rgb)
        plt.title('Stop Sign Detection')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        # cv2.imwrite("output_image_with_detection.jpg", output_image)

    return ("stop sign", final_result, stop_sign_mask)

# # example execution
# if __name__ == '__main__':
#     glob = glob.glob('Data/Test/IMG_149*.jpg')
#     for file in glob:
#         print(f"Processing file: {file}")
#         is_present = stop_sign_detected(file, debug=True)
#         print(f"\nStop sign present: {is_present}\n")
# 
# input_img = 'Data/crossing_guard.jpg' # replace with image name
# is_present = stop_sign_detected(input_img, debug=True)