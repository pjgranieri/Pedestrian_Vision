import cv2
import numpy as np
import easyocr
import os
import re
import torch

reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

STREET_SUFFIXES = {'AVE', 'AVENUE', 'ST', 'STREET', 'RD', 'ROAD', 'BLVD', 'BOULEVARD',
                   'LN', 'LANE', 'DR', 'DRIVE', 'WAY', 'CT', 'COURT', 'PL', 'PLACE',
                   'PKWY', 'PARKWAY', 'HWY', 'HIGHWAY', 'CIRCLE', 'CIR'}

def clean_text(text):
    """Remove numbers and clean text"""
    text = re.sub(r'\b\d+\b', '', text)  # Remove numbers
    text = re.sub(r'^[Ll]3?goo\s*', '', text, flags=re.IGNORECASE)
    # Remove OCR errors like "Ed ", "@F ", but preserve directional prefixes N, S, E, W
    text = re.sub(r'^@?[A-DF-MO-RT-VX-Z]d?\s+', '', text)  # Remove "Ed ", "@F ", etc but keep N,S,E,W
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_roi_for_ocr(roi):
    """Enhance ROI image for better OCR results"""
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Use gentler settings to avoid over-processing
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Light denoising only
    denoised = cv2.fastNlMeansDenoising(enhanced, h=5)

    # Convert back to BGR for EasyOCR
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

def extract_all_street_names(ocr_results, roi_x, roi_y, roi_h, debug=False):
    """
    Extract ALL valid street names from OCR results
    Returns with ABSOLUTE coordinates in the full image
    Filters out small text (likely directional info)
    """
    if not ocr_results:
        return []

    # First pass: find the maximum text height in this ROI
    max_text_height = 0
    for bbox, text, conf in ocr_results:
        bbox_np = np.array(bbox)
        bbox_h = int(np.max(bbox_np[:, 1]) - np.min(bbox_np[:, 1]))
        max_text_height = max(max_text_height, bbox_h)

    if debug and max_text_height > 0:
        print(f"      Max text height in ROI: {max_text_height}px")

    valid_names = []

    for bbox, text, conf in ocr_results:
        text = text.strip()
        if len(text) < 2:
            continue

        clean = clean_text(text)
        if not clean:
            continue

        words = clean.upper().split()
        if not words:
            continue

        # Calculate bbox position and size
        bbox_np = np.array(bbox)
        bbox_x = int(np.min(bbox_np[:, 0]))
        bbox_y = int(np.min(bbox_np[:, 1]))
        bbox_w = int(np.max(bbox_np[:, 0]) - bbox_x)
        bbox_h = int(np.max(bbox_np[:, 1]) - bbox_y)

        # Convert to ABSOLUTE coordinates
        abs_x = roi_x + bbox_x
        abs_y = roi_y + bbox_y

        # FILTER: Skip text that's too small (less than 20% of ROI height)
        # This filters out very small artifacts
        if bbox_h < roi_h * 0.2:
            if debug:
                print(f"      ✗ Too small (ROI %): '{clean}' (height={bbox_h}px, ROI={roi_h}px)")
            continue

        # FILTER: Skip text that's much smaller than the largest text in this ROI
        # This filters out directional text like "Oakland" and "East Allegheny"
        # Main street names should be the largest text on the sign
        if max_text_height > 0 and bbox_h < max_text_height * 0.7:
            if debug:
                print(f"      ✗ Too small (relative): '{clean}' (height={bbox_h}px vs max={max_text_height}px)")
            continue
        
        # Check if it looks like a street name
        has_suffix = any(w in STREET_SUFFIXES for w in words)
        suffix_only = len(words) == 1 and has_suffix
        
        # KEEP suffix-only for now (for merging), with LOWER confidence threshold
        if suffix_only:
            if conf > 0.3:  # Lowered from 0.5 to catch more suffixes
                valid_names.append({
                    'text': clean,
                    'conf': conf,
                    'location': (abs_x, abs_y, bbox_w, bbox_h),
                    'has_suffix': has_suffix,
                    'suffix_only': True
                })
                if debug:
                    print(f"      ⚠️  Suffix only (kept for merging): '{clean}' (conf={conf:.1%})")
            elif debug:
                print(f"      ✗ Suffix only (low conf): '{clean}' (conf={conf:.1%})")
            continue
        
        # Accept if:
        # 1. Has suffix (e.g., "Forbes Ave", "Cedar Ave")
        # 2. High confidence single word 5+ letters (e.g., "Bellefield", "Forbes")
        # 3. Multi-word phrase 8+ chars (e.g., "East Allegheny")
        
        is_valid = False
        reason = ""
        
        if has_suffix:
            is_valid = True
            reason = "has suffix"
        elif len(words) == 1 and len(clean) >= 5 and conf > 0.6:
            is_valid = True
            reason = "high conf single word"
        elif len(words) >= 2 and len(clean) >= 8:
            is_valid = True
            reason = "multi-word"
        
        if is_valid and conf > 0.15:
            valid_names.append({
                'text': clean,
                'conf': conf,
                'location': (abs_x, abs_y, bbox_w, bbox_h),
                'has_suffix': has_suffix,
                'suffix_only': False
            })
            if debug:
                print(f"      ✓ Valid: '{clean}' (conf={conf:.1%}, {reason})")
        elif debug:
            print(f"      ✗ Rejected: '{clean}' (conf={conf:.1%})")
    
    return valid_names

def find_street_signs(image_path, debug=False):
    """
    Find street signs - one sign per detection
    Returns: list of signs with locations
    """
    img = cv2.imread(image_path)
    if img is None:
        return [], None
    
    H, W = img.shape[:2]
    if debug:
        print(f"Image: {W}x{H} pixels")
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avg_brightness = np.mean(hsv[..., 2])
    is_night = avg_brightness < 140
    if debug:
        print(f"Brightness: {avg_brightness:.0f} ({'NIGHT' if is_night else 'DAY'} mode)")
    
    # Blue detection
    if is_night:
        # Increased min saturation from 8 to 35 to reduce false positives
        lower = np.array([78, 35, 10])
        upper = np.array([142, 255, 255])
    else:
        lower = np.array([88, 40, 40])
        upper = np.array([132, 255, 255])
    
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if debug:
        print(f"Found {len(contours)} blue regions\n")
    
    signs = []
    
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        # Basic filtering
        if w < 50 or h < 25:
            continue
        if area > 0.25 * H * W:
            continue
        if w > 0.75 * W or h > 0.5 * H:
            continue
        
        ar = w / h
        if ar < 0.5 or ar > 15:
            continue
        
        # Blue fill
        roi_mask = mask[y:y+h, x:x+w]
        blue_ratio = np.count_nonzero(roi_mask) / (w * h)
        if blue_ratio < 0.05:
            continue

        # Extract ROI
        roi = img[y:y+h, x:x+w]

        # Additional saturation filter to reduce false positives
        roi_hsv = hsv[y:y+h, x:x+w]
        avg_roi_sat = np.mean(roi_hsv[..., 1])
        min_sat = 25 if is_night else 35
        if avg_roi_sat < min_sat:
            continue
        
        if debug:
            print(f"ROI {i} [{w}x{h}]:")

        # OCR - get ALL text in this region
        try:
            # Try OCR on original image first
            results = reader.readtext(roi, paragraph=False)

            # If no results, try with preprocessing
            if not results:
                roi_processed = preprocess_roi_for_ocr(roi)
                results = reader.readtext(roi_processed, paragraph=False)
            
            if not results:
                if debug:
                    print("  No text detected\n")
                continue
            
            # Extract ALL valid street names with ABSOLUTE coordinates
            street_names = extract_all_street_names(results, x, y, h, debug=debug)
            
            if not street_names:
                if debug:
                    print("  No valid street names\n")
                continue
            
            # Add each valid street name as a separate sign
            for name_info in street_names:
                street_name = name_info['text']
                conf = name_info['conf']
                
                # Confidence check (lowered for night mode)
                min_conf = 0.08 if is_night else 0.15
                if conf < min_conf:
                    if debug:
                        print(f"  ✗ Low confidence: '{street_name}' {conf:.1%}")
                    continue
                
                if debug:
                    print(f"  ✓ ACCEPTED: '{street_name}' ({conf:.1%})")
                
                signs.append({
                    'location': name_info['location'],
                    'text_clean': street_name,
                    'text': street_name,
                    'confidence': conf,
                    'suffix_only': name_info['suffix_only']
                })
            
            if debug:
                print()
                
        except Exception as e:
            if debug:
                print(f"  Error: {e}\n")
            continue
    
    return signs, img

def merge_street_name_fragments(signs, debug=False):
    """
    Merge nearby signs that together form a complete street name
    Uses adaptive distance and considers both vertical and horizontal alignment
    """
    if len(signs) < 2:
        filtered = [s for s in signs if not s.get('suffix_only', False)]
        if debug and len(filtered) < len(signs):
            print(f"🔗 Removed {len(signs) - len(filtered)} standalone suffix-only signs\n")
        return filtered
    
    if debug:
        print(f"\n🔗 Attempting to merge {len(signs)} signs...")
    
    merged = []
    used = [False] * len(signs)
    
    for i in range(len(signs)):
        if used[i]:
            continue
        
        x1, y1, w1, h1 = signs[i]['location']
        text1 = signs[i]['text_clean']
        is_suffix_only1 = signs[i].get('suffix_only', False)
        
        if debug:
            suffix_marker = " [SUFFIX-ONLY]" if is_suffix_only1 else ""
            print(f"  Sign {i}: '{text1}' @ ({x1},{y1}){suffix_marker}")
        
        # Look for nearby complement
        best_match = None
        best_distance = float('inf')
        
        for j in range(len(signs)):
            if i == j or used[j]:
                continue
            
            x2, y2, w2, h2 = signs[j]['location']
            text2 = signs[j]['text_clean']
            is_suffix_only2 = signs[j].get('suffix_only', False)
            
            # Calculate both distance metrics
            cx1, cy1 = x1 + w1/2, y1 + h1/2
            cx2, cy2 = x2 + w2/2, y2 + h2/2
            distance = ((cx1-cx2)**2 + (cy1-cy2)**2)**0.5
            
            # Calculate horizontal and vertical distances
            h_distance = abs(cx1 - cx2)
            v_distance = abs(cy1 - cy2)
            
            # Check alignment type
            is_vertically_aligned = h_distance < max(w1, w2) * 0.8 and v_distance > h_distance
            is_horizontally_aligned = v_distance < max(h1, h2) * 1.5 and h_distance > v_distance
            
            # Adaptive distance threshold
            if is_vertically_aligned:
                max_distance = 500  # Stacked signs (Bellefield above Ave)
            elif is_horizontally_aligned:
                max_distance = 600  # Side-by-side on same line (Cedar | Ave)
            else:
                max_distance = 250  # Default
            
            if debug and distance < max_distance * 1.2:
                if is_vertically_aligned:
                    align_info = "V-ALIGNED"
                elif is_horizontally_aligned:
                    align_info = "H-ALIGNED"
                else:
                    align_info = "OFFSET"
                print(f"    → Sign {j}: '{text2}' @ ({x2},{y2}), dist={distance:.0f}px ({align_info})")
            
            if distance > max_distance:
                continue
            
            # Check if they complement each other
            if not is_suffix_only1 and is_suffix_only2:
                if distance < best_distance:
                    best_match = j
                    best_distance = distance
                    if debug:
                        print(f"      ✓ Potential match (name + suffix)")
            elif is_suffix_only1 and not is_suffix_only2:
                if distance < best_distance:
                    best_match = j
                    best_distance = distance
                    if debug:
                        print(f"      ✓ Potential match (suffix + name)")
        
        if best_match is not None:
            # Merge them
            text2 = signs[best_match]['text_clean']
            
            # Determine order
            if is_suffix_only1:
                combined = f"{text2} {text1}"  # name + suffix
            else:
                combined = f"{text1} {text2}"  # name + suffix
            
            if debug:
                print(f"    ✅ MERGED: '{text1}' + '{text2}' → '{combined}'")
            
            # Merged location
            x2, y2, w2, h2 = signs[best_match]['location']
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            x_max = max(x1+w1, x2+w2)
            y_max = max(y1+h1, y2+h2)
            
            merged.append({
                'location': (x_min, y_min, x_max-x_min, y_max-y_min),
                'text_clean': combined,
                'text': combined,
                'confidence': max(signs[i]['confidence'], signs[best_match]['confidence']),
                'suffix_only': False
            })
            
            used[i] = True
            used[best_match] = True
        else:
            # Keep as-is, but skip suffix-only that didn't merge
            if not is_suffix_only1:
                merged.append(signs[i])
                used[i] = True
                if debug:
                    print(f"    → No match found, keeping as-is")
            else:
                used[i] = True
                if debug:
                    print(f"    → Suffix-only didn't merge, discarding")
    
    if debug:
        print(f"✅ Merge complete: {len(signs)} → {len(merged)} signs\n")
    
    return merged

def detect_street_signs(image_path, debug=False):
    """
    Main function to detect street signs and return two aligned lists.

    Returns:
        tuple of two lists:
            1. List of tuples: [("street sign", is_present, mask), ...]
               - First element is always the string "street sign" (label)
               - is_present (bool): True if sign found, False otherwise
               - mask (np.array): Binary mask showing sign region (255=sign, 0=background)
                                 Shape: (H, W, 1) - exactly like stop sign detector

            2. List of strings: [street_name1, street_name2, ...]
               - Just the detected street names

        Both lists have the same length and order, so indices match.

    Examples:
        >>> matrix_list, name_list = detect_street_signs("image.jpg")
        >>> # If Bellefield Ave detected first, then Forbes Ave:
        >>> matrix_list = [("street sign", True, mask1), ("street sign", True, mask2)]
        >>> name_list = ["Bellefield Ave", "Forbes Ave"]
        >>>
        >>> # Can iterate together:
        >>> for (label, is_present, mask), street_name in zip(matrix_list, name_list):
        ...     if is_present:
        ...         print(f"{label}: {street_name}")

        Single sign: ([("street sign", True, mask)], ["Forbes Ave"])
        No signs: ([("street sign", False, empty_mask)], [""])
    """
    signs, img = find_street_signs(image_path, debug=debug)

    if img is None:
        return [], []

    H, W = img.shape[:2]

    if not signs:
        # No signs found - return single tuple with empty values in both lists
        if debug:
            print("❌ No signs found\n")
        empty_mask = np.zeros((H, W, 1), dtype=np.uint8)
        return [("street sign", False, empty_mask)], [""]
    
    # Merge fragments
    signs = merge_street_name_fragments(signs, debug=debug)
    
    # Remove low confidence junk
    signs = [s for s in signs if s['confidence'] > 0.3 or 
            any(w in STREET_SUFFIXES for w in s['text_clean'].upper().split())]
    
    # Deduplicate
    final_signs = []
    seen_texts = set()
    for s in sorted(signs, key=lambda x: -x['confidence']):
        text_key = s['text_clean'].upper().replace(' ', '')
        
        is_duplicate = False
        for seen in list(seen_texts):
            if text_key in seen or seen in text_key:
                is_duplicate = True
                if len(text_key) > len(seen):
                    seen_texts.remove(seen)
                    seen_texts.add(text_key)
                    final_signs = [f for f in final_signs if f['text_clean'].upper().replace(' ', '') != seen]
                    final_signs.append(s)
                break
        
        if not is_duplicate:
            final_signs.append(s)
            seen_texts.add(text_key)
    
    signs = final_signs

    if not signs:
        if debug:
            print("❌ No valid signs after filtering\n")
        empty_mask = np.zeros((H, W, 1), dtype=np.uint8)
        return [("street sign", False, empty_mask)], [""]

    # Convert to output format: TWO aligned lists
    matrix_list = []  # List of ("street sign", is_present, mask)
    name_list = []    # List of street names (strings)

    for s in signs:
        x, y, w, h = s['location']
        street_name = s['text_clean']

        # Create binary mask EXACTLY like stop sign detector
        # Shape: (H, W, 1) with dtype=np.uint8
        street_sign_mask = np.zeros((H, W, 1), dtype=np.uint8)
        cv2.rectangle(street_sign_mask, (x, y), (x+w, y+h), (255), -1)

        # Add to matrix list: ("street sign", True, mask)
        matrix_list.append(("street sign", True, street_sign_mask))

        # Add to name list: just the street name string
        name_list.append(street_name)


    if debug:
        print(f"{'='*70}")
        print(f"✅ FOUND {len(matrix_list)} STREET SIGN(S):")
        print(f"{'='*70}\n")

        for i, ((label, is_present, mask), street_name) in enumerate(zip(matrix_list, name_list), 1):
            print(f"  {i}. Matrix: ('{label}', {is_present}, mask shape: {mask.shape})")
            print(f"     Name:   '{street_name}'")

    return matrix_list, name_list

# Main
if __name__ == "__main__":
    images = ["IMG_1473.jpg", "IMG_1478.jpg", "IMG_1480.jpg", "IMG_1490.jpg", "IMG_1492.jpg", "IMG_1496.jpg"]
    
    for img_file in images:
        path = f"Data/Test/{img_file}"
        if not os.path.exists(path):
            continue
        
        print(f"\n{'='*70}")
        print(f"🔍 {img_file}")
        print(f"{'='*70}\n")

        # Get results as two lists
        matrix_list, name_list = detect_street_signs(path, debug=True)

        # Display results
        print(f"{'='*70}")
        print(f"RESULTS:")
        print(f"{'='*70}")
        print(f"\nMatrix List ({len(matrix_list)} items):")
        for i, (label, is_present, mask) in enumerate(matrix_list, 1):
            print(f"{i}. Label: '{label}'")
            print(f"   Present: {is_present}")
            print(f"   Mask: {mask.shape}, pixels={np.count_nonzero(mask)}")
            print()

        print(f"Name List ({len(name_list)} items):")
        for i, street_name in enumerate(name_list, 1):
            print(f"{i}. '{street_name}'")
        print()

        # Optional: Visualize masks
        if matrix_list[0][1]:  # If any sign found
            img = cv2.imread(path)
            if img is not None:
                for (label, is_present, mask), street_name in zip(matrix_list, name_list):
                    if is_present:
                        # Convert 3D mask (H, W, 1) to 2D for contour finding
                        mask_2d = mask.squeeze()

                        # Draw green rectangle where mask is
                        contours, _ = cv2.findContours(mask_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(img, contours, -1, (0, 255, 0), 5)

                        # Get bounding box for text
                        if contours:
                            x, y, w, h = cv2.boundingRect(contours[0])
                            cv2.putText(img, street_name, (x, max(30, y-15)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                out_path = f"Data/{os.path.splitext(img_file)[0]}_DETECTED.jpg"
                cv2.imwrite(out_path, img)
                print(f"💾 Saved visualization: {out_path}\n")