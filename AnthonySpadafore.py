##################################################################################################
# LIBRARIES
##################################################################################################

#IMPORT LIBRARIES
import cv2 #openCV (C++)
import os
import numpy as np
import matplotlib.pyplot as plt

##################################################################################################
# FUNCTIONS
##################################################################################################

#FUNCTION: multiScaleTemplateMatching
def multiScaleTemplateMatching(source, scalars, template):

    found = None
    for scalar in scalars:

        #RESIZE IMAGE
        resized_template = cv2.resize(template, (int(template.shape[1]*scalar), int(template.shape[0]*scalar)))

        #ENSURE TEMPLATE IS SMALLER
        if source.shape[0] < resized_template.shape[0] or source.shape[1] < resized_template.shape[1]:
            continue
            
        #PERFORM TEMPLATE MATCHING
        result = cv2.matchTemplate(source, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        #UPDATE IF BETTER MATCH IS FOUND
        if found is None or max_val > found[0]:
            found = (max_val, max_loc, scalar)
    
    return found


#FUNCTION: stopSignalDetection
def stopSignalDetection(originalHSV, stop_template, isNight, debug):

    #INITIALIZE OUTPUT
    stopMatrix = np.zeros((originalHSV.shape[0], originalHSV.shape[1], 1), dtype=np.uint8)
    stopFound = False

    #RED COLOR THRESHOLDING
    if isNight:
        redLower = (30, 150, 150)
        redHigher = (70, 255, 255)
        mask1 = cv2.inRange(originalHSV, redLower, redHigher)
        redLower = (20, 55, 150)
        redHigher = (80, 95, 255)
        mask2 = cv2.inRange(originalHSV, redLower, redHigher)
        maskRed = cv2.bitwise_or(mask1, mask2) #overall mask
    # elif not isNight:
    #     redLower = (0, 150, 150)
    #     redHigher = (20, 255, 255)
    #     mask1 = cv2.inRange(originalHSV, redLower, redHigher)
    #     redLower = (200, 150, 150)
    #     redHigher = (360, 255, 255)
    #     mask2 = cv2.inRange(originalHSV, redLower, redHigher)
    #     maskRed = cv2.bitwise_or(mask1, mask2) #overall mask

    #CLOSE HOLES
    kernel = np.ones((3, 3), np.uint8)
    maskRed = cv2.morphologyEx(maskRed, cv2.MORPH_CLOSE, kernel, iterations=3) #close holes

    #MULTI-SCALE TEMPLATE MATCHING
    scalars = np.linspace(0.3, 2.0, 20)[::-1] #scalars for resizing template
    found = multiScaleTemplateMatching(maskRed, scalars, stop_template) #function call

    #DRAW RECTANGLE IF FOUND
    (max_val, max_loc, scalar) = found
    if max_val >= 0.70:
        start_x = int(max_loc[0] )
        start_y = int(max_loc[1] )
        end_x = int((max_loc[0] + stop_template.shape[1]*scalar))
        end_y = int((max_loc[1] + stop_template.shape[0]*scalar))
        cv2.rectangle(stopMatrix, (start_x, start_y), (end_x, end_y), 255, -1)
        stopFound = True
        print(f"STOP SIGNAL detected at ({start_x}, {start_y})")
    else:
        print(f"STOP SIGNAL not detected (r = {max_val})")

    #DEBUG
    if debug:
        originalRGB = cv2.cvtColor(originalHSV, cv2.COLOR_HSV2RGB_FULL)
        annotatedImage = originalRGB.copy()
        if max_val >= 0.70:
            cv2.rectangle(annotatedImage, (start_x, start_y), (end_x, end_y), (0, 255, 0), 20)
            plt.figure(figsize=[10, 30])
            plt.subplot(141); plt.imshow(originalRGB); plt.title("Original Image"); plt.axis("off"); plt.tight_layout()
            plt.subplot(142); plt.imshow(maskRed, cmap="gray"); plt.title(f"Red Thresholded Image (r = {max_val})"); plt.axis("off"); plt.tight_layout()
            plt.subplot(143); plt.imshow(annotatedImage, cmap="gray"); plt.title(f"Annotated Image (stopFound = {stopFound})"); plt.axis("off"); plt.tight_layout()
            plt.subplot(144); plt.imshow(stopMatrix, cmap="gray"); plt.title(f"Location Mask (stopFound = {stopFound})"); plt.axis("off"); plt.tight_layout()
            plt.show()

    #RETURN
    return ("stop signal", stopFound, stopMatrix)


#FUNCTION: walkSignalDetection
def walkSignalDetection(originalHSV, walk_template, isNight, debug):

    #INITIALIZE OUTPUT
    walkMatrix = np.zeros((originalHSV.shape[0], originalHSV.shape[1], 1), dtype=np.uint8)
    walkFound = False

    #WHITE COLOR THRESHOLDING
    if isNight:
        whiteLower = (0, 0, 240)
        whiteHigher = (360, 5, 255)
        maskWhite = cv2.inRange(originalHSV, whiteLower, whiteHigher) #overall mask
    # elif not isNight:
    #     whiteLower = (0, 0, 150)
    #     whiteHigher = (360, 10, 255)
    #     maskWhite = cv2.inRange(originalHSV, whiteLower, whiteHigher) #overall mask

    #CLOSE HOLES
    kernel = np.ones((3, 3), np.uint8)
    maskWhite = cv2.morphologyEx(maskWhite, cv2.MORPH_CLOSE, kernel, iterations=2) #close holes

    #MULTI-SCALE TEMPLATE MATCHING
    scalars = np.linspace(0.3, 2.0, 20)[::-1] #scalars for resizing template
    found = multiScaleTemplateMatching(maskWhite, scalars, walk_template) #function call

    #DRAW RECTANGLE IF FOUND
    (max_val, max_loc, scalar) = found
    if max_val >= 0.70:
        start_x = int(max_loc[0] )
        start_y = int(max_loc[1] )
        end_x = int((max_loc[0] + walk_template.shape[1]*scalar))
        end_y = int((max_loc[1] + walk_template.shape[0]*scalar))
        cv2.rectangle(walkMatrix, (start_x, start_y), (end_x, end_y), 255, -1)
        walkFound = True
        print(f"WALK SIGNAL detected at ({start_x}, {start_y})")
    else:
        print(f"WALK SIGNAL not detected (r = {max_val})")

    #DEBUG
    if debug:
        originalRGB = cv2.cvtColor(originalHSV, cv2.COLOR_HSV2RGB_FULL)
        annotatedImage = originalRGB.copy()
        if max_val >= 0.70:
            cv2.rectangle(annotatedImage, (start_x, start_y), (end_x, end_y), (0, 255, 0), 20)
            plt.figure(figsize=[10, 30])
            plt.subplot(141); plt.imshow(originalRGB); plt.title("Original Image"); plt.axis("off"); plt.tight_layout()
            plt.subplot(142); plt.imshow(maskWhite, cmap="gray"); plt.title(f"White Thresholded Image (r = {max_val})"); plt.axis("off"); plt.tight_layout()
            plt.subplot(143); plt.imshow(annotatedImage, cmap="gray"); plt.title(f"Annotated Image (walkFound = {walkFound})"); plt.axis("off"); plt.tight_layout()
            plt.subplot(144); plt.imshow(walkMatrix, cmap="gray"); plt.title(f"Location Mask (walkFound = {walkFound})"); plt.axis("off"); plt.tight_layout()
            plt.show()
            
    #RETURN
    return ("walk signal", walkFound, walkMatrix)


##################################################################################################
# MAIN
##################################################################################################

#READ TEMPLATES
file = os.path.join('Data/Templates', 'WALK_TEMPLATE.jpg')
walk_template = cv2.imread(file)
walk_template = cv2.cvtColor(walk_template, cv2.COLOR_BGR2GRAY)
file = os.path.join('Data/Templates', 'STOP_TEMPLATE.jpg')
stop_template = cv2.imread(file)
stop_template = cv2.cvtColor(stop_template, cv2.COLOR_BGR2GRAY)

#CONVERT TEMPLATES TO BINARY
_, walk_template = cv2.threshold(walk_template, 100, 255, cv2.THRESH_BINARY)
_, stop_template = cv2.threshold(stop_template, 100, 255, cv2.THRESH_BINARY)

#READ IMAGES
original = cv2.imread('Data/Walk Signal/IMG_1481.jpg')
originalHSV = cv2.cvtColor(original, cv2.COLOR_BGR2HSV_FULL) #convert to HSV (0-360, 0-255, 0-255)

#FIND RESULTS
result1 = stopSignalDetection(originalHSV, stop_template, isNight=True, debug=True)
result2 = walkSignalDetection(originalHSV, walk_template, isNight=True, debug=True)
