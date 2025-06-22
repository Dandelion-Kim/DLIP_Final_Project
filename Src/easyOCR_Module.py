###############################################################################
# @course 	Deep Learning and Image Processing - HGU
# @author	ChanJung Kim / 22000188
#           JongHyeon Kim / 21900179
# @Created	2025-06-13 by CJKIM
# @Modified	2025-06-21 by JHKIM
# @brief	[DLIP] Final Project Module which contains easyOCR related function
###############################################################################

import cv2 
import numpy as np
from Levenshtein import ratio
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import easyocr

# These parameters you may change if no letters are detected from ROI at Debug Mode
beta = 40                           # used in convertScaleAbs in lines 174
LaplacianConstant = 6               # used in laplacian kernel in lines 27

# EasyOCR Configuration
reader = easyocr.Reader(['en'])

# Kernel Declaration
kernelMorph2x2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
kernelMorph3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
kernel = np.array([[0, -1, 0], [-1, LaplacianConstant, -1], [0, -1, 0]])

# You can add more IC chip lists here 
IC_INFO = {
    "SN74LS47N": {
        "label": "7-SEGMENT DECODER",
        "pinmap": ["B", "C", "LT", "BI/RBO", "RBI", "D", "A", "GND", "VCC", "f", "g", "a", "b", "c","d","e"]
    },
    "SN74LS74N": {
        "label": "POS-EDGE-TRIGGERED D-FF",
        "pinmap": ["1CLR","1D","1CK","1PR","1Q","1Q_not","GND","VCC","2CLR","2D","2CK","2PR","2Q","2Q_not"] 
    },
    "SN74HC00N": {
        "label": "NAND Gate",
        "pinmap": ["1A", "1B", "1Y", "2A", "2B", "2Y", "GND", "VCC", "4B", "4A", "4Y", "3B", "3A", "3Y"]
    },
    "SN74HC04N": {
        "label": "HEX Inverter",
        "pinmap": ["1A", "1Y", "2A", "2Y", "3A", "3Y", "GND", "VCC", "6A", "6Y", "5A", "5Y", "4A", "4Y"]
    },
    "SN74HC08N": {
        "label": "AND Gate",
        "pinmap": ["1A", "1B", "1Y", "2A", "2B", "2Y", "GND", "VCC", "4B", "4A", "4Y", "3B", "3A", "3Y"]
    },
    "SN74HC86N": {
        "label": "XOR Gate",
        "pinmap": ["1A", "1B", "1Y", "2A", "2B", "2Y", "GND", "VCC", "4B", "4A", "4Y", "3B", "3A", "3Y"]
    },  
    "CD74HC4052E": {
        "label": "H-SPEED CMOS MUX & D-MUX",
        "pinmap": ["B0", "B2", "BN", "B3", "B1", "Enot", "VEE", "GND", "VCC", "A2", "A1", "AN", "A0", "A3", "S0", "S1"]
    },
    "GD74LS74A": {
        "label": "DUAL POS-EDGE-TRIGGERD D-FF",
        "pinmap": ["1CLR", "1D", "1CK", "1PR", "1Q", "1Qnot", "GND", "VCC", "2CLR", "2D", "2CK", "2PR", "2Q", "2Qnot"]
    }
}

# text : Text to replace letter
def Clean(text):
    return text.replace('B', '8')\
               .replace('O', '0')\
               .replace('Z', '7')\
               .replace('I', '1')\
               .replace("SN74", "")\
               .replace("SN", "")

# frame : Image to rotate
# angle : The angle of rotation
def RotateImage(frame: cv2.typing.MatLike,
                angle: float):
    h,w = frame.shape[:2]
    center = (w//2,h//2)

    # Rotates a given image by a given angle with respect to a given center point.
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotatedImg = cv2.warpAffine(frame, M,(w,h))

    return rotatedImg, M

# frame : Image to find contours
def FindingContours(frame: cv2.typing.MatLike):
    # Color Convert: BGR -> GRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Pre-Processing & Fillter
    mask = cv2.adaptiveThreshold(gray,272,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,167,34)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelMorph2x2,iterations=2)
    mask = cv2.morphologyEx(mask,cv2.MORPH_DILATE,kernelMorph2x2,iterations=1)

    # Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

# frame     : Original image(main)
# chipList  : Detected IC Chip lists
# contours  : ROI contours information
# flag      : List that contain mode flag information
def IdentifyICchip(frame: cv2.typing.MatLike, 
                chipList: list, 
                contours: cv2.typing.MatLike,
                flag: list):
    # Variable to Return Initiation
    point = []
    length = []
    whRatio = []
    minAreaRectWH = []
    theta = []
    boxData = []
    idx = 0
    flagCount = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:  # Exception for Too small contour

            ########################################################################################
            ############################### Needed Information Diriven #############################
            ########################################################################################
            x, y, w, h = cv2.boundingRect(cnt)
            # Identify if Chip lays as horizontal or vetical
            if w > h:
                Horizontal = True
            else:
                Horizontal = False
            # ROI Selection
            roi = frame[y:y+h, x:x+w]

            # Finding rectangle points and angle with minimum area
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            _, (wc, hc), angle = rect
            angleROI = angle
            angleGUI = angle

            # Calculate ratio of width and height 
            rate = wc/hc
            if wc < hc:
                rate = hc/wc
                angleROI = angle - 90

            ########################################################################################
            ##################################### Pre-Processing ###################################
            ########################################################################################
            if rate > 2.4 and rate < 3.2:
                cv2.drawContours(frame,[box],0,(0,255,0),2)     # Draw a box on the frame.

                # Angle compensation
                if wc > hc and Horizontal == False:
                    angleGUI -= 90
                if Horizontal == True:
                    angleGUI -= 90
                    if wc < hc:
                        angleGUI -= 90

                # Align ROI Horizontally
                roi, _ = RotateImage(roi, angleROI)
                M2 = cv2.getRotationMatrix2D((frame.shape[1]//2, frame.shape[0]//2), angleGUI, 1.0)  

                # Calculate Rotated Box Coordinate for Future Use
                box = np.array(box, dtype=np.float32).reshape(-1, 1, 2)
                box = cv2.transform(box, M2).reshape(-1, 2)
            
                # Extracting OCR Input Images
                roiOriginal = roi.copy()
                roiBright = cv2.convertScaleAbs(roiOriginal, alpha=1, beta=beta)
                roiOriLaplacian = cv2.filter2D(roiOriginal, -1, kernel)
                roiBriLaplacian = cv2.filter2D(roiBright, -1, kernel)

                # Resize ROI to Increase Validity  
                if wc > hc:
                    resize_y = 200 / (hc)
                else :
                    resize_y = 200 / wc
                roiOriginal = cv2.resize(roiOriginal, None, fx = resize_y, fy = resize_y, interpolation=cv2.INTER_CUBIC)
                roiBright = cv2.resize(roiBright, None, fx = resize_y, fy = resize_y, interpolation=cv2.INTER_CUBIC)
                roiOriLaplacian = cv2.resize(roiOriLaplacian, None, fx = resize_y, fy = resize_y, interpolation=cv2.INTER_CUBIC)
                roiBriLaplacian = cv2.resize(roiBriLaplacian, None, fx = resize_y, fy = resize_y, interpolation=cv2.INTER_CUBIC)

                ##########################################################################################
                #################################### Identify Letters ####################################
                ##########################################################################################
                result1 = reader.readtext(roiOriginal)
                result2 = reader.readtext(roiBright)
                result3 = reader.readtext(roiOriLaplacian)
                result4 = reader.readtext(roiBriLaplacian)

                # Sum-up all the Detected Text
                allTexts=[]
                for item in result1+result2+result3+result4:
                    allTexts.append(item[1])

                # List to store bestvmatch candidates
                bestTrueCandidates = []

                # Evaluate Detected Text
                for text in allTexts:
                    matches = [(true, ratio(Clean(text.upper()), Clean(true))) for true in IC_INFO]
                    best_match = max(matches, key=lambda x: x[1])  
                    bestTrueCandidates.append(best_match)

                # Find Highest Score from bestTrueCandidates
                if bestTrueCandidates:

                    final_best = max(bestTrueCandidates, key=lambda x: x[1]) 
                   
                    # Threshold of Accuracy for Better Confidential Result
                    if final_best[1] > 0.55:
                        chipList[idx] = final_best
                        flagCount=1
                        if not flag[4]:
                            print(f"Detected: {final_best[0]} with accuracy of {final_best[1]*100:.2f}%")

                # Debug Mode!
                if (flag[3]):
                    cv2.imshow("ROI Original",  roiOriginal)
                    cv2.imshow("ROI Bright",    roiBright)
                    cv2.imshow("ROI Ori Lap",   roiOriLaplacian)
                    cv2.imshow("ROI Bri Lap",   roiBriLaplacian)

                #####################################################################################
                #################################### DATA Stored ####################################
                #####################################################################################
                point.append((x, y))
                length.append((w, h))
                whRatio.append(rate)
                minAreaRectWH.append((wc, hc))
                theta.append(angleGUI)
                boxData.append(box.tolist())
                idx += 1

    return frame, point, length, whRatio, minAreaRectWH, theta, boxData, flagCount

# commandKey    : Key input
# frame         : Original image(main)
# cap           : VideoCapture Informaion
# modeFlag      : List that contain mode flag information
def ModeChanger(commandKey: str, 
                frame: cv2.typing.MatLike,
                cap: cv2.VideoCapture, 
                modeFlag: list):

    # Function: Pause, Resume, and Save 
    if commandKey == ord('p') or commandKey == ord('P'):
        commandKey = 0
        print("Pause Mode")
        while (True):
            Key_2 = cv2.waitKey(30)

            # Save Function
            if Key_2 == ord('s') or Key_2 == ord('S'):
                filename = input("FileName:")
                cv2.imwrite(filename+".png",frame)
                print(f"Saved: {filename}.png")

            # Resume Function
            elif Key_2 == ord('r') or Key_2 == ord('R'):
                print("Resume!")
                break

            # Exit
            elif Key_2 == 27:
                print("ByeBye!!")
                cap.release() 
                exit()

    # Mode 1: Default Mode
    elif commandKey == ord('1'):
        print("Current Mode: Mode 1")
        modeFlag[0] = False
        modeFlag[1] = False

    # Mode 2: Node Identification
    elif commandKey == ord('2'):
        if not modeFlag[0]:
            print("Mode-On: Mode 2")
        else:
            print("Release Mode 2")
        modeFlag[0] = not modeFlag[0]

    # Mode 3: Internal Circuit Display
    elif commandKey == ord('3'):
        if not modeFlag[1]:
            print("Mode-On: Mode 3")
        else:
            print("Release Mode 3")        
        modeFlag[1] = not modeFlag[1]

    # Mode 4: Data-Sheet
    elif commandKey == ord('4'):
        print("Corresponding data-sheet will be poped up!")
        modeFlag[2] = True

    # Mode 5: Debug Mode
    elif commandKey == ord('5'):
        if not modeFlag[3]:
            print("Debug Mode!!")
        else:
            print("Release Debug Mode")
            # Destroy ROI Windowss
            cv2.destroyWindow("ROI Original")
            cv2.destroyWindow("ROI Bright")
            cv2.destroyWindow("ROI Ori Lap")
            cv2.destroyWindow("ROI Bri Lap")
        modeFlag[3] = not modeFlag[3]

    # Exit Command
    elif commandKey == 27:
        print("ByeBye!!")
        cap.release() 
        exit()

    return modeFlag

# pinMaps   : Pin mapping information
# frame     : Original image(main)
# x         : x point
# y         : y point
# w         : Width
# h         : Height
def PinMapDraw(pinMaps: list, 
               frame: cv2.typing.MatLike,
               x: float,
               y: float,
               w: float,
               h: float):
    
    # Calculate Distance between Nodes
    offset = h / (((int)(len(pinMaps)/2))+1)
    for j in range((int)(len(pinMaps)/2)):
        # Assign Text Printing Coordinates
        (text_width, _), _ = cv2.getTextSize(pinMaps[j], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        margin = 10                             # Empty space margun between text and roi
        text_x = x - text_width - margin
        text_x2 = x + w + margin
        text_y = int(y + offset * (j + 1))

        if text_x < 0:                          # Prevent if text_x points locate out of frame
            text_x = 0

        # Display texts side and side.
        cv2.putText(frame, pinMaps[j], (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, pinMaps[j+(int)(len(pinMaps)/2)], (int(text_x2), int(y + offset * (j + 1))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# frame     : Original image(main)
# chipList  : Detected IC Chip lists
# rectPoint : Square enclosing ROI points coordinate in original IMG
# rectWH    : Square enclosing ROI width and height in original IMG
# whMinRect : ROI width and height
# rectRatio : ROI width and height ration
# boxPoint  : ROI 4 points coordinates 
# ang       : Angle of ROI
# modeFlag  : List that contain mode flag information
def GUIPrint (frame: cv2.typing.MatLike, 
              chipList: list, 
              rectPoint: list, 
              rectWH: list, 
              whMinRect: list, 
              rectRatio: list, 
              boxPoint: list, 
              ang: list, 
              modeFlag: list):
    
    idx = 0
    for i in range(len(rectPoint)):
        # Make Empty Frame for Node and Internal Circuit Display
        EmptyFrame = np.zeros(frame.shape, np.uint8)

        if rectRatio[i] > 2.4 and rectRatio[i] < 3.2 and chipList[idx][0] != '':
            # Data Unpackaging
            chip = chipList[idx][0]
            idx += 1
            chipText = (chip, IC_INFO[chip]["label"])
            x, y, w, h = rectPoint[i][0], rectPoint[i][1], rectWH[i][0], rectWH[i][1]
            wc, hc = whMinRect[i][0], whMinRect[i][1]
            pinMap = IC_INFO[chip]["pinmap"]
            BoxTemp = boxPoint[i]

            # Make Rotation Matrix for Node and Internal Circuit Display
            M = cv2.getRotationMatrix2D((frame.shape[1]//2, frame.shape[0]//2), ang[i], 1.0)
    
            smallest_two = sorted(BoxTemp, key=lambda p: p[0])[:2]  # Only two small x-point box coordinates were received. 
            smallest = sorted(smallest_two, key=lambda p:p[1])[:1]  # Then, only one of the smallest box coordinates in y-point is received.

            # Conpensate for PinMapDraw
            if (wc > hc):
                wc, hc = hc, wc

            # Mode 2 / 3 / 4 
            if modeFlag[0] or modeFlag[1] or modeFlag[2]:
                
                # Mode 2 - Node pin mapping.
                if modeFlag[0]:     
                    PinMapDraw(pinMap, EmptyFrame, smallest[0][0], smallest[0][1], wc, hc)
                
                # Mode 3 - Display circuit diagram image.
                if modeFlag[1]:     
                    ChipMap = cv2.imread("img/"+chip+".png")     # Use relative path
                    if ChipMap is not None:
                        resized_ChipMap = cv2.resize(ChipMap,(int(wc),int(hc)))

                        # Make an Exception for when Size of resized_ChipMap does not fit
                        x = int(smallest[0][0]); y = int(smallest[0][1])
                        h_frame, w_frame = EmptyFrame.shape[:2]
                        if y + int(hc) <= h_frame and x + int(wc) <= w_frame:
                            EmptyFrame[y:y+int(hc), x:x+int(wc)] = resized_ChipMap
                    else:
                        print(f"Circuit Images Not Found 404")
                
                # Mode 4 - Open PDF datasheet.
                if modeFlag[2]:     
                    pdf_path = os.path.join("pdf", chip+".pdf")  # Use relative path
                    if os.path.exists(pdf_path):
                        os.startfile(pdf_path)
                    else:
                        print("PDF 파일이 존재하지 않습니다:", pdf_path)

                # Print IC Name for Mode 2, 3, 4.
                for k in range(len(chipText)):
                    size, _ = cv2.getTextSize(chipText[k], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    dx = size[0]/2
                    dy = size[1]/2 + (size[1] + size[1]/3) * (len(chipText) - k)
                    cv2.putText(EmptyFrame,chipText[k],((int)(smallest[0][0] + wc/2 - dx),(int)(smallest[0][1]-dy)),cv2.FONT_HERSHEY_SIMPLEX,0.6, (36, 255, 12), 2)
                
                # Rotate the Frame Back to its Original Position.
                M_inv = cv2.invertAffineTransform(M)
                EmptyFrame = cv2.warpAffine(EmptyFrame, M_inv, (frame.shape[1], frame.shape[0]))
                # Mask and Print out Shaped Letters and Images in the Original Image.
                mask = cv2.cvtColor(EmptyFrame, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                cv2.copyTo(EmptyFrame, mask, frame)
            else:
                # Print IC Name for Mode 1.
                for k in range(len(chipText)):  
                    size, _ = cv2.getTextSize(chipText[k], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    dx = size[0]/2
                    dy = size[1]/2 + (size[1] + size[1]/3) * (len(chipText) - k)
                    cv2.putText(frame,chipText[k],((int)(x + w/2 - dx),(int)(y - dy)),cv2.FONT_HERSHEY_SIMPLEX,0.6, (36, 255, 12), 2)
    
    modeFlag[2] = False
