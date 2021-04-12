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
import random
from typing import List
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd


tiff_image = cv2.imread("Composite_z001_c001.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
image = cv2.normalize(tiff_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#cv2.imshow("Image", image)

#gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# bilateral filter
bilateral = cv2.bilateralFilter(image,3,1,1)

# median filter
median = cv2.medianBlur(image,3)
#cv2.imshow("median", median)

# gauss filter to denoise
gauss = cv2.GaussianBlur(image, (3, 3), 0)

# edge detection
#median = np.uint8(median)
edged = cv2.Canny(median, 50, 120)
#cv2.imshow("edge", edged)

# fill up the gap within objects
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

#cv2.imshow("edge_filled", edged)


# keypoints
sift = cv2.xfeatures2d.SIFT_create()

kp = sift.detect(gauss,None)

img=cv2.drawKeypoints(gauss,kp,outImage=None)

cv2.imshow('sift_keypoints.jpg',img)
cv2.imshow('Image',image)

cv2.waitKey(0)

# find objects
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)


def point_connect(pots: List[list]):
    l = len(pots)
    if l <= 1:
        return [], 0
    # points that already linked
    con = [pots[0]]   
    # points that not linked
    not_con = pots[1:] 
    # all the lines
    paths = []         
    # total length
    length_total = 0   
   
    for _ in range(l - 1): 
        # caculate the distance between a and b
        # chose random 2 points
        a, b = con[0], not_con[0] 
        length_ab = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        for m in con:
            for n in not_con:
                lg = math.sqrt((m[0] - n[0]) ** 2 + (m[1] - n[1]) ** 2)
                # if there is shorter distance
                if lg < length_ab: 
                    length_ab = lg
                    a, b = m, n
 
        
        paths.append([pots.index(a), pots.index(b)])  
        con.append(b)      
        not_con.remove(b)  
        length_total += length_ab  
 
    return paths, length_total


paths = []
total_length = []
key_point = []

for c in cnts:
        # if it's too samll, it might be noise, just ignore it
    if cv2.contourArea(c) < 30:
        continue

    for kp in c:
       key_point.append(kp)

def flat(l):
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            yield from flat(k)

for i in key_point:
    print((list(flat(key_point[i].tolist()))))

point = image.copy()
for i in key_point:
    cv2.circle(point, (list(flat(key_point[1].tolist()))), 1, (0,0,255), -1)

cv2.imshow('point',point)
cv2.waitKey(0)
