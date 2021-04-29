import os
os.environ['DISPLAY'] = ':1'
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from PIL import Image
import numpy as np
import argparse
import imutils
import glob
import pandas as pd
import cv2
from cv2 import drawContours
import math
from typing import List
from skimage import filters

#Reload the saved prediction images
predict_score_images = [] 
for directory_path_pre in glob.glob("Semantic_Segmentation/training_data/test_predict_output/"):
    predict_score_images_path = glob.glob(os.path.join(directory_path_pre, '*.png'))
    predict_score_images_path.sort()
    for k in predict_score_images_path:
        predict_score_image = cv2.imread(k, 0)
        predict_score_images.append(predict_score_image)
        #train_labels.append(label)
#Convert list to array          
predict_score_images = np.array(predict_score_images)
print("Testset predictions quantity: " + str(len(predict_score_images)))

#Capture testset mask/label info as a list
score_masks = [] 
for directory_path_lab in glob.glob("Semantic_Segmentation/training_data/label_test"):
    score_mask_path = glob.glob(os.path.join(directory_path_lab, '*.png'))
    score_mask_path.sort()
    for m in score_mask_path:
        score_mask = cv2.imread(m, 0)
        score_reflect_mask = cv2.copyMakeBorder(score_mask,8,8,8,8,cv2.BORDER_REFLECT)       
        score_masks.append(score_reflect_mask)
#Convert list to array           
score_masks = np.array(score_masks)
print("Testset images quantity: " + str(len(score_masks)))

#Capture testset images info as a list
score_images = [] 
for directory_path_img in glob.glob("Semantic_Segmentation/training_data/image_test"):
    score_image_path = glob.glob(os.path.join(directory_path_img, '*.png'))
    score_image_path.sort()
    for n in score_image_path:
        score_image = cv2.imread(n, 0)
        score_reflect_image = cv2.copyMakeBorder(score_image,8,8,8,8,cv2.BORDER_REFLECT)       
        score_images.append(score_reflect_image)
#Convert list to array           
score_images = np.array(score_images)
print("Testset images quantity: " + str(len(score_masks)))


#Measure the lengths of Microtubules in testset

count = 1

for image in predict_score_images:

    #image = np.array(image, dtype=np.uint8)
    #Connect separate parts
    #dilataion = cv2.dilate(image, None, iterations=1)
    #erosion = cv2.erode(dilataion, None, iterations=1)

    label_original = score_masks[count-1]
    testset_original = score_images[count-1]

    #gray = cv2.cvtColor(erosion,cv2.COLOR_BGR2GRAY)
    # threshold
    #thresh = cv2.threshold(image,128,255,cv2.THRESH_BINARY)[1]

    # convert to binary by thresholding
    ret, binary_map = cv2.threshold(image,127,255,0)

    #binary_map = cv2.normalize(binary_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)

    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]

    image_noise_reduce = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if areas[i] >= 80:   #keep
            image_noise_reduce[labels == i + 1] = 255

    #image_noise_reduce_copy = image_noise_reduce.copy()
    # get contours
    cnts = cv2.findContours(image_noise_reduce, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    pixelsPerMetric = None

    # center point
    def midpoint(ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    boxes = []
    tltrX_list = []
    tltrY_list = []
    blbrX_list = []
    blbrY_list = []

    tlblX_list = []
    tlblY_list = []
    trbrX_list = []
    trbrY_list = []

    dA_list = []
    dB_list = []

    # for loop for all contour
    for c in cnts:
        # if it's too small, it might be noise, just ignore it
        if cv2.contourArea(c) < 20:
            continue

        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        
        box = perspective.order_points(box)

        boxes.append(box.astype("int"))

        #cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        #for (x, y) in box:
            #cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        tltrX_list.append(tltrX)
        tltrY_list.append(tltrY)
        blbrX_list.append(blbrX)
        blbrY_list.append(blbrY)

        tlblX_list.append(tlblX)
        tlblY_list.append(tlblY)
        trbrX_list.append(trbrX)
        trbrY_list.append(trbrY)


        #cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        #cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        #cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        #cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        #cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
        #cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)

        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        dA_list.append(dA)
        dB_list.append(dB)

        #if pixelsPerMetric is None:
        #	pixelsPerMetric = dB / imgsize[1]

        #dimA = dA / pixelsPerMetric
        #dimB = dB / pixelsPerMetric

        #cv2.putText(orig, "{:.1f}".format(dA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        #cv2.putText(orig, "{:.1f}".format(dB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)

        #cv2.imshow("Image", orig)
        #cv2.waitKey(0)

    tltrX_list = np.array(tltrX_list)
    tltrY_list = np.array(tltrY_list)
    blbrX_list = np.array(blbrX_list)
    blbrY_list = np.array(blbrY_list)

    tlblX_list = np.array(tlblX_list)
    tlblY_list = np.array(tlblY_list)
    trbrX_list = np.array(trbrX_list)
    trbrY_list = np.array(trbrY_list)

    dA_list = np.array(dA_list)
    #print(dA_list,dB_list)

    cnts_number = len(dA_list)
    print('The predicted image %s of testset detects Microtubules: %d' %(count, cnts_number))

    img = image_noise_reduce.copy()
    for i in range(len(tltrX_list)):
        cv2.line(img, (int(tltrX_list[i]), int(tltrY_list[i])), (int(blbrX_list[i]), int(blbrY_list[i])),(255, 0, 255), 2)
        cv2.line(img, (int(tlblX_list[i]), int(tlblY_list[i])), (int(trbrX_list[i]), int(trbrY_list[i])),(255, 0, 255), 2)
        cv2.putText(img, "{:.1f}".format(dA_list[i]), (int(tltrX_list[i] - 15), int(tltrY_list[i] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(img, "{:.1f}".format(dB_list[i]), (int(trbrX_list[i] + 10), int(trbrY_list[i])), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)

        cv2.imshow("Image", img)

    img_box = image.copy()
    for k in range(len(boxes)):
        cv2.drawContours(img_box, [boxes[k].astype("int")], -1, (0, 255, 0), 2)

        #cv2.imshow("Image_box", img_box)

    cv2.waitKey(3000)
    count = count + 1