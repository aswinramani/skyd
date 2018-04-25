import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def visualize(image):
    plt.rcParams['figure.figsize'] = (7,7)
    plt.imshow(image, cmap="gray")
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def rotate(template, degree):
    rows,cols = template.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), degree,1)
    r_template = cv2.warpAffine(template,M,template.shape[::-1])
    return r_template

#  /** object detection using template matching **/
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
            # pts = list(zip(*loc[::-1])) / for python 2.7
            pts = list(zip(*loc[::-1]))
            threshold = round(threshold - mod, 2)
        result[r] = round(threshold + mod, 2)
    return result
            
def detect(img, template, display=False):
    w, h = template.shape[::-1]
    image_copy = img.copy()
    img_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    r_list = [0,45,90,135,180]
    result = verify(img_gray, template, r_list) 
    # best_rotation_deg = dict((key,value) for key, value in result.iteritems() if value == max(result.values())) # py2.7
    best_rotation_deg = dict((key,value) for key, value in result.items() if value == max(result.values()))
    if len(best_rotation_deg) > 1:
        from collections import OrderedDict
        best_rotation_deg = OrderedDict(sorted(best_rotation_deg.items(), key=lambda t: t[0]))  
    keys = list(best_rotation_deg.keys())
    values = list(best_rotation_deg.values()) 
    # res = cv2.matchTemplate(img_gray,rotate(template, best_rotation_deg.keys()[0]),cv2.TM_CCORR_NORMED)# py2.7
    res = cv2.matchTemplate(img_gray,rotate(template, keys[0]),cv2.TM_CCORR_NORMED)
    # loc = np.where(res >= best_rotation_deg.values()[0]) # py2.7
    loc = np.where(res >= values[0])
    # pts = zip(*loc[::-1]) # py2.7
    pts = list(zip(*loc[::-1]))
    for pt in pts:
        bounding_box = cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (255,0,0), 1)
    if display:
        visualize(img)
    return pts

def dev():
    size = (w, h, channels) = (20, 20, 3)
    template_image = np.zeros(size, dtype=np.uint8)
    template_image[4:6, 4:18] = [255, 255, 255]
    template_image[4:18, 4:6] = [255, 255, 255]
    # template_image[4:18, 4:6] = [255, 255, 255]
    t_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    w, h = t_gray.shape[::-1]
    # visualize(t_gray)
    img = cv2.imread("images/DSC01798.JPG")
    image_copy = img.copy()
    # Step 3 Rescale Image (here we downsample our image)
    down1 = cv2.pyrDown(image_copy)
    img_gray = cv2.cvtColor(down1, cv2.COLOR_BGR2GRAY)
    r_list = [0,45,90,135,180,225,270,315]
    result = {}
    for r in r_list:
        res = cv2.matchTemplate(img_gray,rotate(t_gray, r),cv2.TM_CCORR_NORMED)
        threshold = np.max(res) - np.std(res)
        result[r] = threshold
    # import code
    # code.interact(local=dict(globals(), **locals()))
    # from collections import OrderedDict
    # sorted_results = OrderedDict(sorted(result.items(), key=lambda t: t[1]))
    # for result in sorted_results:

    # best_rotation_deg = dict((key,value) for key, value in result.iteritems() if value == max(result.values()))
    # res = cv2.matchTemplate(img_gray,rotate(t_gray, best_rotation_deg.keys()[0]),cv2.TM_CCORR_NORMED)
    # loc = np.where(res >= best_rotation_deg.values()[0])
    # pts = zip(*loc[::-1])
    # for pt in pts:
    #     cv2.rectangle(down1, pt, (pt[0] + w, pt[1] + h), (255,0,0), 1)

    # visualize(down1)
def run():
    # dev()
    src_dir = "images/"
    names = []
    locations = []
    count = 0
    # file_path = os.path.join(directory, file_name)
    # imageList = ["DSC01798.JPG"]
    imageList = filter(lambda x: x.split(".")[1] == "JPG", os.listdir(src_dir))
    for image in imageList:
        # Step 1 Read source image in bgr 
        img = cv2.imread(src_dir + image)
        # Step 2 Add template image for searching L shaped objects 
        template = get_template_image()
        # create a copy of the source image 
        image_copy = img.copy()
        # Image Pyramids
        # Step 3 Rescale Image (here we downsample our image)
        down1 = cv2.pyrDown(image_copy)
        # Step 4 Detect object using template matching
        # points = map(lambda x: [x[0],x[1]], detect(down1, template)) # for py2.7
        points = list(map(lambda x: [x[0],x[1]], detect(down1, template))) # for py3.6
        names.append(image)
        locations.append(points)

    df = pd.DataFrame(data={"Location": locations}, index=names)
    df.index.name = "FileNames"
    output_dir = "detections"
    output_file_name = "positions.csv"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    file_path = os.path.join(output_dir, output_file_name)
    df.to_csv(file_path, encoding='utf-8')

if __name__ == '__main__':
    run()


# Debugger Code
# import code
# code.interact(local=dict(globals(), **locals()))