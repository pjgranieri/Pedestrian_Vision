#!/usr/bin/env python3
"""
Test suite for sign_finder.py
Tests street sign detection against known ground truth labels
"""

import os
from PJGranieri import detect_street_signs
import cv2
import numpy as np

# Ground truth data: {image_filename: [expected_street_names]}
GROUND_TRUTH = {
    "IMG_4108.jpg": ["Forbes Ave"],
    "IMG_4110.jpg": ["Forbes Ave"],
    "IMG_1473.jpg": ["Fifth Ave"],
    "IMG_1480.JPG": ["S Bouquet St"],
    "IMG_1482.JPG": ["S Bouquet St"],
    "IMG_1483.JPG": ["S Bouquet St", "Forbes Ave"],
    "IMG_1484.JPG": ["Forbes Ave"],
    "IMG_1487.JPG": ["Oakland Ave"],
    "IMG_1489.JPG": ["Atwood St"],
    "IMG_1490.JPG": ["Atwood St"],
    "IMG_1493.JPG": ["Oakland Ave"],
    "IMG_1494.JPG": ["S Bouquet St"],
    "IMG_1498.JPG": ["Forbes Ave", "Bigelow Blvd Schenley Dr"],
    "upright_street_signs.png": ["Bellefield Ave", "Cedar Ave"],
}

def normalize_street_name(name):
    """Normalize street name for comparison (case-insensitive, remove extra spaces)"""
    return ' '.join(name.upper().split())

def compare_results(detected_names, expected_names):
    """
    Compare detected street names with expected names
    
    Returns:
        dict with keys:
            - 'status': 'PASS', 'PARTIAL', or 'FAIL'
            - 'detected': list of detected names
            - 'expected': list of expected names
            - 'correct': list of correctly detected names
            - 'missing': list of expected names that were not detected
            - 'extra': list of detected names that were not expected
    """
    # Normalize all names
    detected_norm = [normalize_street_name(name) for name in detected_names]
    expected_norm = [normalize_street_name(name) for name in expected_names]
    
    # Find matches
    correct = []
    missing = []
    extra = []
    
    for exp in expected_norm:
        found = False
        for det in detected_norm:
            # Check if they match (exact or one contains the other)
            if exp == det or exp in det or det in exp:
                if det not in correct:
                    correct.append(det)
                found = True
                break
        if not found:
            missing.append(exp)
    
    # Find extra detections
    for det in detected_norm:
        matched = False
        for exp in expected_norm:
            if exp == det or exp in det or det in exp:
                matched = True
                break
        if not matched and det not in extra:
            extra.append(det)
    
    # Determine status
    if len(correct) == len(expected_norm) and len(extra) == 0:
        status = 'PASS'
    elif len(correct) > 0:
        status = 'PARTIAL'
    else:
        status = 'FAIL'
    
    return {
        'status': status,
        'detected': detected_names,
        'expected': expected_names,
        'correct': correct,
        'missing': missing,
        'extra': extra
    }

def run_tests(debug=False, data_dir="Data"):
    """
    Run all tests and return results
    
    Args:
        debug: If True, print detailed debug info for each image
        data_dir: Directory containing test images
    
    Returns:
        dict with test statistics
    """
    results = {}
    total_tests = len(GROUND_TRUTH)
    passed = 0
    partial = 0
    failed = 0
    
    print("=" * 80)
    print("STREET SIGN DETECTION TEST SUITE")
    print("=" * 80)
    print()
    
    for img_file, expected_names in GROUND_TRUTH.items():
        img_path = os.path.join(data_dir, img_file)
        
        # Check if file exists
        if not os.path.exists(img_path):
            print(f"⚠️  SKIPPED: {img_file} (file not found)")
            print()
            results[img_file] = {
                'status': 'SKIPPED',
                'detected': [],
                'expected': expected_names,
                'correct': [],
                'missing': expected_names,
                'extra': []
            }
            total_tests -= 1
            continue
        
        print(f"Testing: {img_file}")
        print("-" * 80)
        
        # Run detection
        try:
            detections = detect_street_signs(img_path, debug=debug)
            
            # Extract street names from results
            detected_names = []
            for sign_tuple in detections:
                name, is_present, mask = sign_tuple
                if is_present and name:  # Only count if present and not empty
                    detected_names.append(name)
            
            # Compare with ground truth
            result = compare_results(detected_names, expected_names)
            results[img_file] = result
            
            # Print result
            status_symbol = {
                'PASS': '✅',
                'PARTIAL': '⚠️ ',
                'FAIL': '❌'
            }
            
            print(f"Status: {status_symbol[result['status']]} {result['status']}")
            print(f"Expected: {expected_names}")
            print(f"Detected: {detected_names}")
            
            if result['correct']:
                print(f"Correct:  {result['correct']}")
            if result['missing']:
                print(f"Missing:  {result['missing']}")
            if result['extra']:
                print(f"Extra:    {result['extra']}")
            
            # Update counters
            if result['status'] == 'PASS':
                passed += 1
            elif result['status'] == 'PARTIAL':
                partial += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            results[img_file] = {
                'status': 'ERROR',
                'detected': [],
                'expected': expected_names,
                'correct': [],
                'missing': expected_names,
                'extra': [],
                'error': str(e)
            }
            failed += 1
        
        print()
    
    # Print summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests:    {total_tests}")
    print(f"✅ Passed:      {passed} ({100*passed/total_tests if total_tests > 0 else 0:.1f}%)")
    print(f"⚠️  Partial:     {partial} ({100*partial/total_tests if total_tests > 0 else 0:.1f}%)")
    print(f"❌ Failed:      {failed} ({100*failed/total_tests if total_tests > 0 else 0:.1f}%)")
    print("=" * 80)
    print()
    
    # Detailed breakdown
    if partial > 0 or failed > 0:
        print("DETAILED BREAKDOWN:")
        print("-" * 80)
        for img_file, result in results.items():
            if result['status'] in ['PARTIAL', 'FAIL', 'ERROR']:
                print(f"\n{img_file}: {result['status']}")
                print(f"  Expected: {result['expected']}")
                print(f"  Detected: {result['detected']}")
                if result.get('missing'):
                    print(f"  Missing:  {result['missing']}")
                if result.get('extra'):
                    print(f"  Extra:    {result['extra']}")
                if result.get('error'):
                    print(f"  Error:    {result['error']}")
        print()
    
    return {
        'total': total_tests,
        'passed': passed,
        'partial': partial,
        'failed': failed,
        'results': results
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test street sign detection')
    parser.add_argument('--debug', action='store_true', 
                       help='Print detailed debug info for each image')
    parser.add_argument('--data-dir', type=str, default='Data',
                       help='Directory containing test images (default: Data)')
    parser.add_argument('--image', type=str, 
                       help='Test only a specific image')
    
    args = parser.parse_args()
    
    if args.image:
        # Test single image
        if args.image in GROUND_TRUTH:
            img_path = os.path.join(args.data_dir, args.image)
            expected = GROUND_TRUTH[args.image]
            
            print(f"Testing: {args.image}")
            print(f"Expected: {expected}")
            print()
            
            detections = detect_street_signs(img_path, debug=True)
            detected_names = [name for name, is_present, _ in detections if is_present and name]
            
            result = compare_results(detected_names, expected)
            print()
            print(f"Result: {result['status']}")
            print(f"Detected: {detected_names}")
            if result['correct']:
                print(f"Correct: {result['correct']}")
            if result['missing']:
                print(f"Missing: {result['missing']}")
            if result['extra']:
                print(f"Extra: {result['extra']}")
        else:
            print(f"Error: {args.image} not in ground truth data")
    else:
        # Run all tests
        run_tests(debug=args.debug, data_dir=args.data_dir)
