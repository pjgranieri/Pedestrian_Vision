##################################################################################################
# LIBRARIES
##################################################################################################

#IMPORT FILES
import RickyZhao as RZ
import AnthonySpadafore as AS
import PJGranieri as PJ
import ZacharyMcPherson as ZM
import ManelinRajan as MR
from printResults import printResults

#IMPORT LIBRARIES
import cv2 #openCV (C++)
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import easyocr
import torch

#CHECK FOR USEABLE GPU
print("---------------------------------------------------------------------------")
print(f"GPU detected = {torch.cuda.is_available()}")

#INITIALIZE OCR READER
try:
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())  # initialize EasyOCR reader | no GPU
except Exception as e:
    exit()


##################################################################################################
# READ FILES
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

#READ FOLDER
path = "Data/Test"
files = []
for filename in os.listdir(path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        files.append(filename)


##################################################################################################
# TESTING
##################################################################################################

#CREATE result_list
result_list = []

#LOOP THROUGH EACH FILE
for filename in files:

    #FIND IMAGE PATH
    file = os.path.join(path, filename)
    print("---------------------------------------------------------------------------")
    print(f"IMAGE: {file}")

    #RESET result_list
    result_list.clear()

    #READ IMAGE
    original = cv2.imread(file)
    originalRGB = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    originalHSV = cv2.cvtColor(original, cv2.COLOR_BGR2HSV_FULL) #convert to HSV (0-360, 0-255, 0-255)

    #STOP AND WALK SIGNAL DETECTION
    print("---------------------------------------------------------------------------")
    print("FUNCTION: stopSignalDetection")
    ASresult1 = AS.stopSignalDetection(originalHSV, stop_template, isNight=True, debug=False)
    print("---------------------------------------------------------------------------")
    print("FUNCTION: walkSignalDetection")
    ASresult2 = AS.walkSignalDetection(originalHSV, walk_template, isNight=True, debug=False)
    
    #STOP SIGN DETECTION
    print("---------------------------------------------------------------------------")
    print("FUNCTION: stop_sign_detected")
    RZresult = RZ.stop_sign_detected(file, debug=False)

    #STREET SIGN DETECTION
    print("---------------------------------------------------------------------------")
    print("FUNCTION: detect_street_signs")
    PJresultTuples, PJresultNames = PJ.detect_street_signs(file, debug=False)    

    #STREET SIGN DETECTION
    print("---------------------------------------------------------------------------")
    print("FUNCTION: detect_crosswalk")
    MRresult= MR.detect_crosswalk(file, debug=False)    

    # #DISPLAY DETECTION MASKS
    # plt.figure(figsize=[10, 30])
    # plt.subplot(321); plt.imshow(ASresult1[2], cmap='gray'); plt.title(f'Stop Signal Mask (found = {ASresult1[1]})'); plt.axis('off')
    # plt.subplot(322); plt.imshow(ASresult2[2], cmap='gray'); plt.title(f'Walk Signal Mask (found = {ASresult2[1]})'); plt.axis('off')
    # plt.subplot(323); plt.imshow(RZresult[2], cmap='gray'); plt.title(f'Stop Sign Mask (found = {RZresult[1]})'); plt.axis('off')
    # plt.subplot(324); plt.imshow(PJresultTuples[0][2], cmap='gray'); plt.title(f'Street Sign Mask (found = {PJresultTuples[0][1]})'); plt.axis('off')
    # plt.subplot(325); plt.imshow(MRresult[2], cmap='gray'); plt.title(f'Crosswalk Mask (found = {MRresult[1]})'); plt.axis('off')
    # plt.show()

    #CREATE result_list
    result_list.append(ASresult1)
    result_list.append(ASresult2)
    result_list.append(RZresult)
    for tup in PJresultTuples:
        result_list.append(tup)
    result_list.append(MRresult)
    
    #PASS result_list TO findDepth
    print("---------------------------------------------------------------------------")
    print("FUNCTION: findDepth")
    ZMResult = ZM.findDepth(originalRGB, result_list, "DPT_Large", debug=True)

    #CALL printResults
    printResults(ZMResult, PJresultNames)