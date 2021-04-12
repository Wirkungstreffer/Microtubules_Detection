import os
os.environ['DISPLAY'] = ':1'
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from PIL import Image
import numpy as np
import argparse
import imutils
import cv2
from cv2 import drawContours
import math
from typing import List
from skimage import filters
from networkx.algorithms.smallworld import sigma


array_of_img = [] # this if for store all of the image data
# this function is for read image,the input is directory name
def read_directory(directory):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(r"./"+ directory):
        #print(filename) #just for test
        #img is used to store the image data 
        tiff_image = cv2.imread(directory + "/" + filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        img = cv2.normalize(tiff_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        array_of_img.append(img)
        #cv2.imshow("Image", img)
        #cv2.waitKey(0)
        #print(img)
        print(array_of_img)

#Series_Tiff_Files = read_directory("Tiff_files")

# center point
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def centroid(img):
    moments = cv2.moments(img, False)

    cX = float(moments["m10"] / moments["m00"])
    cY = float(moments["m01"] / moments["m00"])

    return [cX, cY]

# read imgae
tiff_image = cv2.imread("Composite_z001_c001.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
image = cv2.normalize(tiff_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
orig = image.copy()

tiff_image_2 = cv2.imread("Composite_z001_c002.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
image_2 = cv2.normalize(tiff_image_2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
orig_2 = image_2.copy()

test = cv2.imread("Test.png")
print(test.shape)

# gray grade
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# bilateral filter
bilateral = cv2.bilateralFilter(image,3,1,1)

bilateral_2 = cv2.bilateralFilter(image_2,3,1,1)

# median filter
median = cv2.medianBlur(image,3)

median_2 = cv2.medianBlur(image_2,3)
#cv2.imshow("median", median)

# gauss filter to denoise
gauss = cv2.GaussianBlur(image, (5, 5), 0)
gauss_copy = gauss.copy()

gauss_2 = cv2.GaussianBlur(image_2, (5, 5), 0)
# NLM
#nlm = cv2.fastNlMeansDenoising(image, None, 10 ,10,7,21)

edged = cv2.Canny(gauss, 20, 50)

edged_2 = cv2.Canny(gauss_2, 30, 120)	
#edged = filters.sobel(gauss)
#cv2.imshow("edge", edged)
#cv2.waitKey(0)

# fill up the gap within objects
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

edged_copy = edged.copy()

edged_2 = cv2.dilate(edged_2, None, iterations=1)
edged_2 = cv2.erode(edged_2, None, iterations=1)

# find Harris corners
image_32 = np.float32(image)
harris = cv2.cornerHarris(image_32,2,3,0.04)

#result is dilated for marking the corners
harris = cv2.dilate(harris,None)
cv2.imshow('Harris',harris)
cv2.waitKey(0)

# Threshold for an optimal value
#image[dst>0.01*dst.max()]=[0,0,255]

# Shi-Tomasi corner detect
corners = cv2.goodFeaturesToTrack(gauss, 50, 0.01, 10)
corners = np.int0(corners)   

for i in corners:
    x, y = i.ravel()
    cv2.circle(gauss, (x, y), 4, (0, 0, 0), -1)
    
cv2.imshow('Shi-Tomasi', gauss)

test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
gray_copy = test_gray.copy()
seed = (2, 2)

cv2.floodFill(test_gray, None, seedPoint=seed, newVal=(0, 0, 0), loDiff=(5), upDiff=(5))
cv2.circle(test_gray, seed, 2, (0, 0, 0), cv2.FILLED, cv2.LINE_AA);

cv2.imshow('flood', test_gray)
cv2.imshow('test', gray_copy)
cv2.waitKey(0)
