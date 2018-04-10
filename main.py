import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas

def visualize(image):
    plt.rcParams['figure.figsize'] = (7,7)
    plt.imshow(image, cmap="gray")
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def harris_corner_detection(image_copy):
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
    # Detect Corners using Harris Corner 
    gray = np.float32(gray)
    corner = cv2.cornerHarris(gray, 2, 3, 0.04)
    corner = cv2.dilate(corner, None)
    # plt.imshow(corner, cmap="gray")
    thresh = 0.1 * corner.max()
    corner_img = np.copy(cpy)
    for j in range(0, corner.shape[0]):
        for i in range(0, corner.shape[1]):
            if(corner[j,i] > thresh):      
                cv2.circle(corner_img, (i,j), 2, (0, 255, 0), 1)
    return corner
    # plt.imshow(corner_img)
    # visualize(corner_img)

def shi_thomas(image_copy):
    gray = cv2.cvtColor(image_copy,cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray,5,0.75,2)
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(image_copy,(x,y),3,255,-1)  
    # visualize(image_copy)
    return corners

def hough_line(image_copy):
    low_threshold = 50
    high_threshold = 100
    i_gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(i_gray, low_threshold, high_threshold)
    # plt.imshow(edges, cmap='gray')
    rho = 1
    theta = np.pi/180
    threshold = 60
    min_line_length = 30
    max_line_gap = 5
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    hough_image = np.copy(image_copy)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(hough_image, (x1,y1), (x2,y2), (255,0,0), 5)
    # visualize(hough_image)
    return lines

def canny_edge_detection(image_copy):
    # rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    lower_threshold = 190
    upper_threshold = 255
    edges = cv2.Canny(gray, lower_threshold, upper_threshold)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 7))
    ax1.set_title('org')
    ax1.imshow(image_copy, cmap="gray")
    ax2.set_title('edges')
    ax2.imshow(edges, cmap="gray")
    plt.show()
    # return edges
    # visualize(edges)

def contour_segmentation(image_copy):
    # rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    # ret1, binary = cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)
    ret1, binary = cv2.threshold(gray,125,255,0)
    # visualize(binary)
    # # find and draw shapes
    ret2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_copy2 = np.copy(image_copy)
    # corners = cv2.goodFeaturesToTrack(gray,25,0.5,2)
    # corners = np.int32(corners)
    coutour_l = []
    for contour in contours: 
        peri = cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
        if len(approx) >= 6:
            coutour_l.append(contour)  
    #         # for i in corners: 
    #         #     for c in contour:   
    #         #         corner_x, corner_y = i.ravel()
    #         #         contour_x, contour_y = c.ravel()
    #         #         if corner_x == contour_x:
    #         #             coutour_l.append(contour)
    # print "total contours detected " + str(len(contours))
    # print "reduced contours after corner detection " + str(len(coutour_l)) 
    all_contours = cv2.drawContours(image_copy2, coutour_l,  -1, (0,255,0), 2)
    # visualize(image_copy2)
    return all_contours

def run():
    image = cv2.imread('images/DSC02209.JPG')
    image_copy = np.copy(image)
    contours = contour_segmentation(image_copy)
    # all_contours = cv2.drawContours(image_copy, contours,  -1, (0,255,0), 2)
    visualize(contours)

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
#     # import code
#     # code.interact(local=dict(globals(), **locals()))
#     ret2, match_contour, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     # print len(match_contour)
#     return match_contour
# def find_gcp(image):

#     # step 1  - convert BGR image to RBG
#     rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     # step 2 - scale our image 
#     max_dim = max(rgb.shape)

# Debugger Code
# import code
# code.interact(local=dict(globals(), **locals()))