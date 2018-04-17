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

def draw_rect(img, pts, w, h):
    for pt in pts:
        bounding_box = cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (255,0,0), 1)

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
            
def detect(img, template, display=False):
    w, h = template.shape[::-1]
    image_copy = img.copy()
    img_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    r_list = [0,45,90,135,180,225,270,315]
    result = verify(img_gray, template, r_list) 
    best_rotation_deg = dict((key,value) for key, value in result.iteritems() if value == max(result.values()))
    if len(best_rotation_deg) > 1:
        from collections import OrderedDict
        best_rotation_deg = OrderedDict(sorted(best_rotation_deg.items(), key=lambda t: t[0]))    
    res = cv2.matchTemplate(img_gray,rotate(template, best_rotation_deg.keys()[0]),cv2.TM_CCORR_NORMED)
    loc = np.where(res >= best_rotation_deg.values()[0])
    pts = zip(*loc[::-1])
    if display:
        draw_rect(img, pts, w, h)
        visualize(img)
    return pts

def run():
    src_dir = "images/"
    names = []
    locations = []
    image_list = filter(lambda x: x.split(".")[1] == "JPG", os.listdir(src_dir))
    for image in image_list:
        # Step 1 Read source image in bgr 
        img = cv2.imread(src_dir + image)
        # Step 2 Add template image for searching L shaped objects 
        template = get_template_image()
        w,h = template.shape[::-1] 
        # create a copy of the source image 
        image_copy = img.copy()
        # Image Pyramids
        # Step 3 Rescale Image (here we downscale our image)
        down1 = cv2.pyrDown(image_copy)
        # Step 4 Detect object using template matching
        down1_points = detect(down1, template)
        # Step 5 Compute locations for source image
        points = map(lambda x: ((2*x[0] + (2*x[0] + w))/2,(2*x[1] + (2*x[1] + h))/2), down1_points)
        names.append(image)
        locations.append(points)

    df = pd.DataFrame(data={"Location": locations}, index=names)
    df.index.name = "FileName"
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