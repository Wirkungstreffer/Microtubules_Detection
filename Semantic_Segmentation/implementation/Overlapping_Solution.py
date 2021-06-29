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
import heapq

# If cv2.imshow() function dosen't work, use the followed line of code
os.environ['DISPLAY'] = ':1'

# Get center point
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
#image = Image.open("Semantic_Segmentation/implementation/prediction_seed/seed_predict_0.png")
#image = image.filter(ImageFilter.ModeFilter(size=13))
#image.save("Semantic_Segmentation/implementation/prediction_seed/seed_predict_0_aa.png")

# seed_overlapping_test = cv2.imread("Semantic_Segmentation/implementation/prediction_seed/test_overlapping.png",0)
image = cv2.imread("Semantic_Segmentation/implementation/prediction_seed/200818_Xb_Reaction2_6uM002_seeds_001_prediction.png")

image_copy = image.copy()

image_grey = image[:,:,0]
# Connect separate parts
seed_dilataion = cv2.dilate(image_grey, None, iterations=1)
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

# Create list for seeds endpoints
seed_endpoints_list = []

seed_boxes = []

seed_tltrX_list = []
seed_tltrY_list = []
seed_blbrX_list = []
seed_blbrY_list = []

seed_tlblX_list = []
seed_tlblY_list = []
seed_trbrX_list = []
seed_trbrY_list = []

seed_dA_list = []
seed_dB_list = []

# for loop for all contour
for seed_c in seed_cnts:
    # if it's too small, it might be noise, just ignore it
    if cv2.contourArea(seed_c) < 5:
        continue

    epsilon = 0.15*cv2.arcLength(seed_c,True)
    approx = cv2.approxPolyDP(seed_c,epsilon,True)

    if (len(approx) >= 3) & (cv2.contourArea(seed_c) > 200):
        
        cv2.drawContours(image_copy, seed_c, -1, (0, 255, 0), 3)
        cv2.drawContours(image_copy, approx, -1, (0, 0, 255), 3)

        cv2.imshow('img',image_copy)
        cv2.waitKey(0)

        #print(len(approx))

        overlapping_coordinate_list = []

        for coordinate in approx:
            overlapping_coordinate_list.append(coordinate[0])

        stuff = list(np.arange(0,len(overlapping_coordinate_list)))
        subset_list = []
        for L in range(0, len(stuff)+1):
            for subset in itertools.combinations(stuff, L):
                if len(subset) == 2:
                    subset_list.append(subset)

        #print(overlapping_coordinate_list)
        #print(overlapping_coordinate_list[subset_list[0][0]])

        different_pixels_list = []

        for sub in range(len(subset_list)):
            img_orig = image.copy()
            cv2.line(img_orig , (tuple(overlapping_coordinate_list[subset_list[sub][0]])), tuple((overlapping_coordinate_list[subset_list[sub][1]])),(255, 255, 255), 3)
            print(tuple(overlapping_coordinate_list[subset_list[sub][0]]),tuple(overlapping_coordinate_list[subset_list[sub][1]]))
            sub_img = cv2.subtract(img_orig, image)
            sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
            non_zero_pixel = cv2.countNonZero(sub_img)
            different_pixels_list.append(non_zero_pixel)

        print(different_pixels_list)

        pair_number = round(len(approx)/2)

        small_number = heapq.nsmallest(pair_number, different_pixels_list) 
        small_index = []
        for t in small_number:
            index = different_pixels_list.index(t)
            small_index.append(index)
            different_pixels_list[index] = 0

        #print(small_number)
        #print(small_index)

        for min_index in small_index:
            seed_tltrX_list.append(overlapping_coordinate_list[subset_list[min_index][0]][0])
            seed_tltrY_list.append(overlapping_coordinate_list[subset_list[min_index][0]][1])
            seed_tlblX_list.append(overlapping_coordinate_list[subset_list[min_index][0]][0])
            seed_tlblY_list.append(overlapping_coordinate_list[subset_list[min_index][0]][1])
            seed_blbrX_list.append(overlapping_coordinate_list[subset_list[min_index][1]][0])
            seed_blbrY_list.append(overlapping_coordinate_list[subset_list[min_index][1]][1])
            seed_trbrX_list.append(overlapping_coordinate_list[subset_list[min_index][1]][0])
            seed_trbrY_list.append(overlapping_coordinate_list[subset_list[min_index][1]][1])

        for min_index_endpoint in small_index:
            seed_endpoints_list.append([(overlapping_coordinate_list[subset_list[min_index_endpoint][0]][0], overlapping_coordinate_list[subset_list[min_index_endpoint][0]][1]),(overlapping_coordinate_list[subset_list[min_index_endpoint][1]][0], overlapping_coordinate_list[subset_list[min_index_endpoint][1]][1])])
        #cv2.imwrite("Semantic_Segmentation/implementation/Overlapping_Solution.png",image_copy)

    else:

        # Use minimal area rectangular to box the segmentation
        seed_box = cv2.minAreaRect(seed_c)
        seed_box = cv2.cv.BoxPoints(seed_box) if imutils.is_cv2() else cv2.boxPoints(seed_box)
        seed_box = np.array(seed_box, dtype="int")
        seed_box = perspective.order_points(seed_box)

        seed_boxes.append(seed_box.astype("int"))

        # Get the midpoint of the length and width of the box
        (seed_tl, seed_tr, seed_br, seed_bl) = seed_box
        
        # Midpoints of the width sides of box
        (seed_tltrX, seed_tltrY) = midpoint(seed_tl, seed_tr)
        (seed_blbrX, seed_blbrY) = midpoint(seed_bl, seed_br)

        # Midpoints of the length sides of box
        (seed_tlblX, seed_tlblY) = midpoint(seed_tl, seed_bl)
        (seed_trbrX, seed_trbrY) = midpoint(seed_tr, seed_br)

        # Add the width sides midpoints into the lists
        seed_tltrX_list.append(seed_tltrX)
        seed_tltrY_list.append(seed_tltrY)
        seed_blbrX_list.append(seed_blbrX)
        seed_blbrY_list.append(seed_blbrY)

        # Add the length sides midpoints into the lists
        seed_tlblX_list.append(seed_tlblX)
        seed_tlblY_list.append(seed_tlblY)
        seed_trbrX_list.append(seed_trbrX)
        seed_trbrY_list.append(seed_trbrY)

        # Calculate the length and width distances
        seed_dA = dist.euclidean((seed_tltrX, seed_tltrY), (seed_blbrX, seed_blbrY))
        seed_dB = dist.euclidean((seed_tlblX, seed_tlblY), (seed_trbrX, seed_trbrY))

        # Add the distance information to the lists
        seed_dA_list.append(seed_dA)
        seed_dB_list.append(seed_dB)

        # Add the endpoints coordinates into list
        if seed_dB >= seed_dA:
            seed_endpoints_list.append([(seed_tlblX, seed_tlblY),(seed_trbrX, seed_trbrY)])
        else:
            seed_endpoints_list.append([(seed_tltrX, seed_tltrY),(seed_blbrX, seed_blbrY)])



print(len(seed_blbrX_list))
print(len(seed_endpoints_list))