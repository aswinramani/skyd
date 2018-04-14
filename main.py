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

def rotate(template, degree):
    rows,cols = template.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), degree,1)
    r_template = cv2.warpAffine(template,M,template.shape[::-1])
    return r_template

# /** object detection using template matching **/
def get_template_image():
    size = (w, h, channels) = (20, 20, 3)
    template_image = np.zeros(size, dtype=np.uint8)
    template_image[4:6, 5:18] = [255, 255, 255]
    template_image[4:18, 4:6] = [255, 255, 255]
    return cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

def get_matching_methods():
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    return methods

def verify(img_gray, template, r_list):
    result = {}
    for r in r_list:
        res = cv2.matchTemplate(img_gray,rotate(template, r),cv2.TM_CCORR_NORMED)
        pts = []
        threshold = 0.8
        mod = 0.02
        while len(pts) < 3:
            loc = np.where(res >= threshold)
            pts = zip(*loc[::-1])
            threshold = round(threshold - mod, 2)
        result[r] = round(threshold + mod, 2)
    return result
            
def detect(img, template):
    w, h = template.shape[::-1]
    image_copy = img.copy()
    img_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    r_list = [0,45,90,135,180]
    result = verify(img_gray, template, r_list) 
    best_rotation_deg = dict((key,value) for key, value in result.iteritems() if value == max(result.values()))
    if len(best_rotation_deg) > 1:
        from collections import OrderedDict
        best_rotation_deg = OrderedDict(sorted(best_rotation_deg.items(), key=lambda t: t[0]))    
    res = cv2.matchTemplate(img_gray,rotate(template, best_rotation_deg.keys()[0]),cv2.TM_CCORR_NORMED)
    loc = np.where(res >= best_rotation_deg.values()[0])
    pts = zip(*loc[::-1])
    # print best_rotation_deg
    # print best_rotation_deg.values()[0]
    # res = cv2.matchTemplate(img_gray,template,cv2.TM_CCORR_NORMED) 
    # threshold = 0.8
    # pts = []
    # while len(pts) < 1:
    #     loc = np.where(res >= threshold)
    #     pts = zip(*loc[::-1])
    #     threshold -= 0.02
    # print threshold
    print len(pts)
    for pt in pts:
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (255,0,0), 1)
    # print pts
    return pts

def run():
    # Read source image in bgr 
    img = cv2.imread('images/DSC02426.JPG')
    # get template image for searching gcp
    template = get_template_image()
    rows,cols = template.shape
    # M = cv2.getRotationMatrix2D((cols/2,rows/2),0,1)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),135,1)
    r_template = cv2.warpAffine(template,M,template.shape[::-1])
    # visualize(r_template)
    # create a copy of the source image 
    image_copy = img.copy()
    # Image Pyramids
    # up1 = cv2.pyrUp(image_copy)
    down1 = cv2.pyrDown(image_copy)
    # down2 = cv2.pyrDown(down1)
    # # down3 = cv2.pyrDown(down2)
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