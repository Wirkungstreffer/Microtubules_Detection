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
import matplotlib.pyplot as plt
from skimage import data, filters, color, morphology
from skimage.segmentation import flood, flood_fill


# center point
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def centroid(img):
    moments = cv2.moments(img, False)

    cX = float(moments["m10"] / moments["m00"])
    cY = float(moments["m01"] / moments["m00"])

    return [cX, cY]

# read imgae
tiff_image = cv2.imread("Region_Segmentation/Composite_z001_c001.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
image = cv2.normalize(tiff_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

tiff_image_2 = cv2.imread("Region_Segmentation/Composite_z001_c002.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
image_2 = cv2.normalize(tiff_image_2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#imgsize = image.shape
#print(imgsize)

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

gauss_2 = cv2.GaussianBlur(image_2, (5, 5), 0)
# NLM
#nlm = cv2.fastNlMeansDenoising(image, None, 10 ,10,7,21)


# closing
closing_2 = cv2.dilate(gauss_2, None, iterations=1)
closing_2 = cv2.erode(closing_2, None, iterations=1)
#cv2.imshow("closing", closing_2)

# edge detection
#median = np.uint8(median)
edged = cv2.Canny(gauss, 20, 50)

edged_2 = cv2.Canny(gauss_2, 30, 120)	
#edged = filters.sobel(gauss)
#cv2.imshow("edge", edged)
#cv2.waitKey(0)

# fill up the gap within objects
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

edged_2 = cv2.dilate(edged_2, None, iterations=1)
edged_2 = cv2.erode(edged_2, None, iterations=1)
#cv2.imshow("edge_filled", edged)

# find objects
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

cnts_2 = cv2.findContours(edged_2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts_2 = imutils.grab_contours(cnts_2)

# order the sequence from left to right of the object
if len(cnts) > 0:
    cnts = contours.sort_contours(cnts)[0]

if len(cnts_2) > 0:
    cnts_2 = contours.sort_contours(cnts_2)[0]
#(cnts, _) = contours.sort_contours(cnts)

pixelsPerMetric = None

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

boxes_2 = []
tltrX_list_2 = []
tltrY_list_2 = []
blbrX_list_2 = []
blbrY_list_2 = []

tlblX_list_2 = []
tlblY_list_2 = []
trbrX_list_2 = []
trbrY_list_2 = []

dA_list_2 = []
dB_list_2 = []
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

#dA_list = np.array(dA_list)
#print(dA_list,dB_list)

# for loop for all contour
for c_2 in cnts_2:
	# if it's too small, it might be noise, just ignore it
	if cv2.contourArea(c_2) < 20:
		continue

	orig = image_2.copy()
	box_2 = cv2.minAreaRect(c_2)
	box_2 = cv2.cv.BoxPoints(box_2) if imutils.is_cv2() else cv2.boxPoints(box_2)
	box_2 = np.array(box_2, dtype="int")
	
	box_2 = perspective.order_points(box_2)

	boxes_2.append(box_2.astype("int"))

	#cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

	#for (x, y) in box:
		#cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

	(tl_2, tr_2, br_2, bl_2) = box_2
	(tltrX_2, tltrY_2) = midpoint(tl_2, tr_2)
	(blbrX_2, blbrY_2) = midpoint(bl_2, br_2)

	(tlblX_2, tlblY_2) = midpoint(tl_2, bl_2)
	(trbrX_2, trbrY_2) = midpoint(tr_2, br_2)

	tltrX_list_2.append(tltrX_2)
	tltrY_list_2.append(tltrY_2)
	blbrX_list_2.append(blbrX_2)
	blbrY_list_2.append(blbrY_2)

	tlblX_list_2.append(tlblX_2)
	tlblY_list_2.append(tlblY_2)
	trbrX_list_2.append(trbrX_2)
	trbrY_list_2.append(trbrY_2)


	#cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	#cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	#cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	#cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

	#cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
	#cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)

	dA_2 = dist.euclidean((tltrX_2, tltrY_2), (blbrX_2, blbrY_2))
	dB_2 = dist.euclidean((tlblX_2, tlblY_2), (trbrX_2, trbrY_2))

	dA_list_2.append(dA_2)
	dB_list_2.append(dB_2)

	#if pixelsPerMetric is None:
	#	pixelsPerMetric = dB / imgsize[1]

	#dimA = dA / pixelsPerMetric
	#dimB = dB / pixelsPerMetric
 
	#cv2.putText(orig, "{:.1f}".format(dA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
	#cv2.putText(orig, "{:.1f}".format(dB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
 
	#cv2.imshow("Image", orig)
	#cv2.waitKey(0)

tltrX_list_2 = np.array(tltrX_list_2)
tltrY_list_2 = np.array(tltrY_list_2)
blbrX_list_2 = np.array(blbrX_list_2)
blbrY_list_2 = np.array(blbrY_list_2)

tlblX_list_2 = np.array(tlblX_list_2)
tlblY_list_2 = np.array(tlblY_list_2)
trbrX_list_2 = np.array(trbrX_list_2)
trbrY_list_2 = np.array(trbrY_list_2)


img = image.copy()
for i in range(len(tltrX_list)):
	cv2.line(img, (int(tltrX_list[i]), int(tltrY_list[i])), (int(blbrX_list[i]), int(blbrY_list[i])),(255, 0, 255), 2)
	cv2.line(img, (int(tlblX_list[i]), int(tlblY_list[i])), (int(trbrX_list[i]), int(trbrY_list[i])),(255, 0, 255), 2)
	cv2.putText(img, "{:.1f}".format(dA_list[i]), (int(tltrX_list[i] - 15), int(tltrY_list[i] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
	cv2.putText(img, "{:.1f}".format(dB_list[i]), (int(trbrX_list[i] + 10), int(trbrY_list[i])), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
#cv2.imshow("Image", img)


img_2 = image_2.copy()
for i_2 in range(len(tltrX_list_2)):
	cv2.line(img, (int(tltrX_list_2[i_2]), int(tltrY_list_2[i_2])), (int(blbrX_list_2[i_2]), int(blbrY_list_2[i_2])),(255, 0, 255), 2)
	cv2.line(img, (int(tlblX_list_2[i_2]), int(tlblY_list_2[i_2])), (int(trbrX_list_2[i_2]), int(trbrY_list_2[i_2])),(255, 0, 255), 2)
	cv2.putText(img, "{:.1f}".format(dA_list_2[i_2]), (int(tltrX_list_2[i_2] - 15), int(tltrY_list_2[i_2] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
	cv2.putText(img, "{:.1f}".format(dB_list_2[i_2]), (int(trbrX_list_2[i_2] + 10), int(trbrY_list_2[i_2])), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
#cv2.imshow("Image", img)

img_box = image.copy()
for k in range(len(boxes)):
    cv2.drawContours(img_box, [boxes[k].astype("int")], -1, (0, 255, 0), 2)
#cv2.imshow("Image_box", img_box)

img_box_2 = image.copy()
for k_2 in range(len(boxes_2)):
    cv2.drawContours(img_box_2, [boxes_2[k_2].astype("int")], -1, (0, 255, 0), 2)
#cv2.imshow("Image_box", img_box)

#caculate contour center
img_endpoints = image.copy()
img_endpoints_2 = image_2.copy()

M_list = []
center_x_list = []
center_y_list = []
count = 0

M_list_2 = []
center_x_list_2 = []
center_y_list_2 = []
count_2 = 0

for p in cnts:
	if cv2.contourArea(p) < 20:
		continue

	M = cv2.moments(p)
	M_list.append(M)
	center_x = int(M["m10"] / M["m00"])
	center_x_list.append(center_x)
	center_y = int(M["m01"] / M["m00"])
	center_y_list.append(center_y)
	count = count+1

for w in range(len(tltrX_list)):
	cv2.circle(img_endpoints, (int(trbrX_list[w]), int(trbrY_list[w])), 7, 128, -1)
	cv2.circle(img_endpoints, (int(tlblX_list[w]), int(tlblY_list[w])), 7, 128, -1)


for p_2 in cnts_2:
	if cv2.contourArea(p_2) < 20:
		continue
	M_2 = cv2.moments(p_2)
	M_list_2.append(M_2)
	center_x_2 = int(M_2["m10"] / M_2["m00"])
	center_x_list_2.append(center_x_2)
	center_y_2 = int(M_2["m01"] / M_2["m00"])
	center_y_list_2.append(center_y_2)
	count_2 = count_2+1

for w_2 in range(len(tltrX_list)):
	cv2.circle(img_endpoints_2, (int(trbrX_list[w_2]), int(trbrY_list[w_2])), 7, 128, -1)
	cv2.circle(img_endpoints_2, (int(tlblX_list[w_2]), int(tlblY_list[w_2])), 7, 128, -1)

cv2.imshow("Image_endpoints", img_endpoints)
cv2.imshow("Image_endpoints_2", img_endpoints_2)







#Flood fill
img_center = image.copy()
img_center_2 = image_2.copy()

gauss_2_copy = gauss_2.copy()
seed = (int(center_x_list_2[1]), int(center_y_list_2[1]))

cv2.floodFill(gauss_2_copy, None, seedPoint=seed, newVal=(255, 255, 255), loDiff=(5), upDiff=(5))
seeds_point = cv2.circle(img_center_2, seed, 2, (0, 0, 0), cv2.FILLED, cv2.LINE_AA)

cv2.imshow('flood_cv', gauss_2_copy)
cv2.imshow('seed point', seeds_point)

#Flood fill in Skimage
gauss_2_copy_2 = gauss_2.copy()

flood_f = flood_fill(gauss_2_copy_2, (int(center_x_list_2[1]), int(center_y_list_2[1])), 255, tolerance=3)
fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

ax[0].imshow(gauss_2_copy_2, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].imshow(flood_f, cmap=plt.cm.gray)
ax[1].plot(int(center_x_list_2[1]), int(center_y_list_2[1]), 'ro') # seed point
ax[1].set_title('After flood fill')
ax[1].axis('off')

plt.show()

# Kmeans color segmentation
#def kmeans_color_quantization(image, clusters=8, rounds=1):
#    h, w = image.shape[:2]
#    samples = np.zeros([h*w,3], dtype=np.float32)
#    count = 0
#
#    for x in range(h):
#        for y in range(w):
#            samples[count] = image[x][y]
#            count += 1
#
#    compactness, labels, centers = cv2.kmeans(samples,
#            clusters, 
#            None,
#            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
#            rounds, 
#            cv2.KMEANS_RANDOM_CENTERS)
#
#    centers = np.uint8(centers)
#    res = centers[labels.flatten()]
#    return res.reshape((image.shape))

# Load image and perform kmeans
#kmeans = kmeans_color_quantization(image_2, clusters=2)
#result = kmeans.copy()

# Floodfill
#seed_point = (int(center_x_list_2[1]), int(center_y_list_2[1]))
#cv2.floodFill(result, None, seedPoint=seed_point, newVal=(36, 255, 12), loDiff=(0, 0, 0, 0), upDiff=(0, 0, 0, 0))

#orig = image.copy()
#cv2.imshow("Image_orig", orig)


class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
 
    def getX(self):
        return self.x
    def getY(self):
        return self.y

def getGrayDiff(img,currentPoint,tmpPoint):
    return abs(int(img[currentPoint.x,currentPoint.y]) - int(img[tmpPoint.x,tmpPoint.y]))

def selectConnects(p):
    #if p != 0:
	connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1), Point(-1, 1), Point(-1, 0)]
    #else:
        #connects = [ Point(0, -1),  Point(1, 0), Point(0, 1), Point(-1, 0)]
	return connects

def regionGrow(img,seeds,thresh,p = 1):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects(p)
    while(len(seedList)>0):
        currentPoint = seedList.pop(0)
 
        seedMark[currentPoint.x,currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
            if grayDiff < thresh and seedMark[tmpX,tmpY] == 0:
                seedMark[tmpX,tmpY] = label
                seedList.append(Point(tmpX,tmpY))
    return seedMark


gauss_2_copy_3 = gauss_2.copy()
#cv2.imshow('gauss_2_copy_3',gauss_2_copy_3)
seeds = [Point(int(center_x_list_2[1]), int(center_y_list_2[1]))]#,Point(int(center_x_list_2[2]), int(center_y_list_2[2])),Point(int(center_x_list_2[3]), int(center_y_list_2[3]))]
RGImg = regionGrow(closing_2, seeds, 3)
cv2.imshow('Region Growing',RGImg)

cv2.waitKey(0)