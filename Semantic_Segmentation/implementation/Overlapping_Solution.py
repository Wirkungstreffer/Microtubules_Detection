import glob
import cv2
from cv2 import drawContours
import os
import datetime
import pydot
os.environ['DISPLAY'] = ':1'
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

#image = Image.open("Semantic_Segmentation/implementation/prediction_seed/seed_predict_0.png")
#image = image.filter(ImageFilter.ModeFilter(size=13))
#image.save("Semantic_Segmentation/implementation/prediction_seed/seed_predict_0_aa.png")

# seed_overlapping_test = cv2.imread("Semantic_Segmentation/implementation/prediction_seed/test_overlapping.png",0)
img = cv2.imread("Semantic_Segmentation/implementation/prediction_seed/seed_predict_0.png")
img_orig = img.copy()
img_copy = img.copy()

img_grey = img[:,:,0]
ret1, thresh = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cnts,hierarchy = cv2.findContours(thresh,2,1)

# for loop for all contour
for c in cnts:
    # if it's too small, it might be noise, just ignore it
    if cv2.contourArea(c) < 20:
        continue

    epsilon = 0.05*cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,epsilon,True)

    cv2.drawContours(img_copy, c, -1, (0, 255, 0), 3)
    cv2.drawContours(img_copy, approx, -1, (0, 0, 255), 3)
    print(approx)

    hull = cv2.convexHull(c,returnPoints = False)
    defects = cv2.convexityDefects(c,hull)

    #cv2.drawContours(img, [hull], -1, (0, 0, 255), 3)

    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(c[s][0])
        end = tuple(c[e][0])
        far = tuple(c[f][0])
        cv2.line(img,start,end,[0,255,0],2)
        cv2.circle(img,far,5,[0,0,255],-1)

#cv2.imshow('img',img)
#cv2.waitKey(0)

fig, ax = plt.subplots(ncols=3, figsize=(20, 20))

ax[0].imshow(img_orig, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].imshow(img, cmap=plt.cm.gray)
ax[1].set_title('hull')
ax[1].axis('off')

ax[2].imshow(img_copy, cmap=plt.cm.gray)
ax[2].set_title('PolyDP')
ax[2].axis('off')

plt.show()