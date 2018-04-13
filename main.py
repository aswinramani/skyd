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

# /** object detection using template matching **/
def get_template_image():
    # size = (w, h, channels) = (10, 10, 3)
    size = (w, h, channels) = (20, 20, 3)
    template_image = np.zeros(size, dtype=np.uint8)
    template_image[4:6, 5:18] = [255, 255, 255]
    template_image[4:18, 4:6] = [255, 255, 255]
    # template_image[2:3, 2:7] = [255, 255, 255]
    # template_image[2:7, 2:3] = [255, 255, 255]
    return cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    # visualize(match_img) cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    # match_copy = np.copy(match_img)
    # match_copy = cv2.rectangle(match_copy,(95, 100),(400,140),(0,255,0),2)
    # match_copy = cv2.rectangle(match_copy,(95, 140),(140,400),(0,255,0),2)
    # visualize(match_copy)
    # gray = cv2.cvtColor(match_copy, cv2.COLOR_BGR2GRAY)
    # ret1, binary = cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)
    # ret1, binary = cv2.threshold(gray,125,255, 0)
    # visualize(binary)
    # ret2, match_contour, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # image_copy2 = np.copy(image_copy)
    # all_contours = cv2.drawContours(match_copy, match_contour,  -1, (0,255,0), 2)
    # visualize(match_copy)
    # print len(match_contour)
    # return all_contours

def get_matching_methods():
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    return methods

def detect(img, template):
    w, h = template.shape[::-1]
    image_copy = img.copy()
    img_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCORR_NORMED) 
    threshold = 0.8
    pts = []
    while len(pts) < 1:
        loc = np.where(res >= threshold)
        pts = zip(*loc[::-1])
        threshold -= 0.1
    print len(pts)
    for pt in pts:
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (255,0,0), 1)
    # import code
    # code.interact(local=dict(globals(), **locals()))
    # visualize(blur)
    return pts

def run():
    # Read source image in bgr 
    img = cv2.imread('images/DSC02209.JPG')
    # get template image for searching gcp
    template = get_template_image()
    # create a copy of the source image 
    image_copy = img.copy()
    # Image Processing
    up1 = cv2.pyrUp(image_copy)
    down1 = cv2.pyrDown(image_copy)
    down2 = cv2.pyrDown(down1)
    # down3 = cv2.pyrDown(down2)
    # Object Detection
    points = detect(down1, template)
    visualize(down1)




if __name__ == '__main__':
    run()


# match_img = np.array([[ 0, 0, 0, 0, 0
#                         0, 255, 0, 0, 0
#                         0, 255, 0, 0, 0
#                         0, 255, 255, 255, 0
#                         0, 0, 0, 0, 0    
                        
# ]])


# Debugger Code
# import code
# code.interact(local=dict(globals(), **locals()))