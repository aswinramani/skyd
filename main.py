import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas

def visualize(image):
    plt.rcParams['figure.figsize'] = (7,7)
    plt.imshow(image, cmap="gray")
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def contour_detection(image_copy):
    gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    ret1, binary = cv2.threshold(gray,125,255,0)
    # find and draw shapes
    ret2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_copy2 = np.copy(image_copy)
    coutour_l = []
    for contour in contours: 
        peri = cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
        # add objects to contour list with vertices greater than or equal to 6
        if len(approx) >= 6:
            coutour_l.append(contour)  
    all_contours = cv2.drawContours(image_copy2, coutour_l,  -1, (0,255,0), 2)
    return binary, contours, all_contours

def run():
    image = cv2.imread('images/DSC02209.JPG')
    image_copy = np.copy(image)
    binary, contours, all_contours = contour_detection(image_copy)
    visualize(all_contours)

if __name__ == '__main__':
    run()

# /** L shape for matching **/
# def match():
#     # size = (w, h, channels) = (50, 50, 3)
#     size = (w, h, channels) = (25, 25, 3)
#     match_img = np.zeros(size, dtype=np.uint8)
#     match_img[4:7, 5:18] = [255, 255, 255]
#     match_img[5:18, 5:7] = [255, 255, 255]
#     match_copy = np.copy(match_img)
#     # match_copy = cv2.rectangle(match_copy,(95, 100),(400,140),(0,255,0),2)
#     # match_copy = cv2.rectangle(match_copy,(95, 140),(140,400),(0,255,0),2)
#     # visualize(match_copy)
#     gray = cv2.cvtColor(match_copy, cv2.COLOR_BGR2GRAY)
#     # gray[95:140, 100:400] = 255
#     # gray[100:400, 100:140] = 255
#     # ret1, binary = cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)
#     ret1, binary = cv2.threshold(gray,125,255, 0)
#     # visualize(binary)
#     ret2, match_contour, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     # print len(match_contour)
#     return match_contour

# Debugger Code
# import code
# code.interact(local=dict(globals(), **locals()))