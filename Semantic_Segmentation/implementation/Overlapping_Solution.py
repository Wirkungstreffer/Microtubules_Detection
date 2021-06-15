import glob
import cv2
from cv2 import drawContours
import os
import datetime
import pydot
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import argparse
import imutils
from scipy import interpolate
import skimage
import skimage.morphology
from skimage.feature import canny
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage import measure
from scipy import ndimage
from matplotlib import cm
import pwlf
import skan
from skimage.morphology import medial_axis, skeletonize
from skimage import morphology
from skan import draw
from skan import skeleton_to_csgraph
from skan import _testdata
import tensorflow as tf
from tensorflow import keras
from fil_finder import FilFinder2D
import astropy.units as u
import pandas as pd
from PIL import Image, ImageFilter
import itertools

# If cv2.imshow() function dosen't work, use the followed line of code
os.environ['DISPLAY'] = ':1'

#image = Image.open("Semantic_Segmentation/implementation/prediction_seed/seed_predict_0.png")
#image = image.filter(ImageFilter.ModeFilter(size=13))
#image.save("Semantic_Segmentation/implementation/prediction_seed/seed_predict_0_aa.png")

# seed_overlapping_test = cv2.imread("Semantic_Segmentation/implementation/prediction_seed/test_overlapping.png",0)
img = cv2.imread("Semantic_Segmentation/implementation/prediction_seed/200818_Xb_Reaction2_6uM002_seeds_001_prediction.png")

img_copy = img.copy()

img_grey = img[:,:,0]
# Connect separate parts
seed_dilataion = cv2.dilate(img_grey, None, iterations=1)
seed_erosion = cv2.erode(seed_dilataion, None, iterations=1)

# Convert prdiction image to binary map by thresholding
seed_ret, seed_binary_map = cv2.threshold(seed_erosion,127,255,0)

# Get the informations of segmentations
seed_nlabels, seed_labels, seed_stats, seed_centroids = cv2.connectedComponentsWithStats(seed_binary_map, None, None, None, 8, cv2.CV_32S)

# Get CC_STAT_AREA component as stats[label, COLUMN] 
seed_areas = seed_stats[1:,cv2.CC_STAT_AREA]

# Create a zero mask to reduce noise
seed_image_noise_reduce = np.zeros((seed_labels.shape), np.uint8)

# Start to reduce noise
for i in range(0, seed_nlabels - 1):
    
    # If the segmented area is large, consider it is not a noise segmentation
    if seed_areas[i] >= 5:   
        seed_image_noise_reduce[seed_labels == i + 1] = 255

# Get contours of segmentations
seed_cnts = cv2.findContours(seed_image_noise_reduce, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
seed_cnts = imutils.grab_contours(seed_cnts)

overlapping_coordinate_list = []
# for loop for all contour
for c in seed_cnts:
    # if it's too small, it might be noise, just ignore it
    if cv2.contourArea(c) < 5:
        continue

    epsilon = 0.15*cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,epsilon,True)

    if len(approx) >= 3:
       
        if cv2.contourArea(c) < 80:
            continue
        
        cv2.drawContours(img_copy, c, -1, (0, 255, 0), 3)
        cv2.drawContours(img_copy, approx, -1, (0, 0, 255), 3)
        for i in approx:
            overlapping_coordinate_list.append(i[0])

        #print(len(approx))


stuff = list(np.arange(0,len(overlapping_coordinate_list)))
subset_list = []
for L in range(0, len(stuff)+1):
    for subset in itertools.combinations(stuff, L):
        if len(subset) == 2:
            subset_list.append(subset)


print(overlapping_coordinate_list[subset_list[0][0]])

for sub in range(len(subset_list)):
    img_orig = img.copy()
    cv2.line(img_orig , (tuple(overlapping_coordinate_list[subset_list[sub][0]])), tuple((overlapping_coordinate_list[subset_list[sub][1]])),(255, 255, 255), 5)
    sub_img = cv2.subtract(img_orig, img)
    sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
    non_zero_pixel = cv2.countNonZero(sub_img)
    print(non_zero_pixel)


cv2.imwrite("Semantic_Segmentation/implementation/Overlapping_Solution.png",img_copy)

#cv2.imshow('img',sub_img)
#cv2.waitKey(0)


