import os
#os.environ['DISPLAY'] = ':1'
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

# Define a class of pixel points coordinate
class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
 
    def getX(self):
        return self.x
    def getY(self):
        return self.y

# Define the growing criterium
def getGrayDiff(img,currentPoint,tmpPoint):
    return abs(int(img[currentPoint.x,currentPoint.y]) - int(img[tmpPoint.x,tmpPoint.y]))

# Define the neighbor connection criterium
def selectConnects(p):
	if p != 0:
	    connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1), Point(-1, 1), Point(-1, 0)]
	else:
		connects = [ Point(0, -1),  Point(1, 0), Point(0, 1), Point(-1, 0)]
	return connects

# Implement region growing
def regionGrow(img,initials,thresh,p = 1):
	# Get the image shape
	height, weight = img.shape

	# Create a blank segmentation space image with input imsize
	intial_Mark = np.zeros(img.shape)
	
	# Create a list to store connected pixel points
	intial_List = []

	# Add up initial points
	for init_point in initials:
		intial_List.append(init_point)

	# Make to be segmented objects as label 1
	label = 1

	# Connecting neighbor pixels
	connects = selectConnects(p)
	while(len(intial_List)>0):
		# Start with initial points
		currentPoint = intial_List.pop(0)
		intial_Mark[currentPoint.x,currentPoint.y] = label
		for i in range(8):
			tmpX = currentPoint.x + connects[i].x
			tmpY = currentPoint.y + connects[i].y
			
			# If there is no pixel fit the criterium or the region grows to the edge of image, stop growing
			if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
				continue
			
			# Calculate the grey difference between seed point and neighbor pixels
			grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))

			# If the difference is smaller than the threshold, add the neighbor to the list
			if grayDiff < thresh and intial_Mark[tmpX,tmpY] == 0:
				intial_Mark[tmpX,tmpY] = label
				intial_List.append(Point(tmpX,tmpY))
	
	# return the marked pixel points list
	return intial_Mark

# Get center point
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

image = cv2.imread("Region_Segmentation/input_image/200818_Xb_Reaction2_6uM003_016.png", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
#image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
gauss = cv2.GaussianBlur(image, (3, 3), 0)

#cv2.imshow('gauss_2_copy_3',gauss_2_copy_3)
intials = [Point(0, 0)]
#RGImg = regionGrow(gauss, intials, 3)
#RGImg = 255*RGImg
#RGImg = 255 - RGImg
#cv2.imwrite("Region_Segmentation/region_growing.png", RGImg)

region_segmentation = cv2.imread("Region_Segmentation/region_growing.png", cv2.IMREAD_GRAYSCALE)

# Connect separate parts
dilataion = cv2.dilate(region_segmentation, None, iterations=1)
erosion = cv2.erode(dilataion, None, iterations=1)

# Convert prdiction image to binary map by thresholding
ret, binary_map = cv2.threshold(erosion,127,255,0)

# Get the informations of segmentations
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)

# Get CC_STAT_AREA component as stats[label, COLUMN] 
areas = stats[1:,cv2.CC_STAT_AREA]

# Create a zero mask to reduce noise
image_noise_reduce = np.zeros((labels.shape), np.uint8)

# Start to reduce noise
for i in range(0, nlabels - 1):
    
    # If the segmented area is large, consider it is not a noise segmentation
    if areas[i] >= 145:   
        image_noise_reduce[labels == i + 1] = 255

cv2.imwrite("Region_Segmentation/region_growing_denoised.png", image_noise_reduce)

# find objects
cnts = cv2.findContours(image_noise_reduce, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#print(len(cnts))

# order the sequence from left to right of the object
if len(cnts) > 0:
    cnts = contours.sort_contours(cnts)[0]

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

endpoints_list = []
microtubules_length = []

# for loop for all contour
for c in cnts:
	# if it's too small, it might be noise, just ignore it
	if cv2.contourArea(c) < 20:
		continue

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

	if dA < dB:
		endpoints_list.append(((tlblX, tlblY), (trbrX, trbrY)))
		microtubules_length.append(dB)
	else:
		endpoints_list.append(((tltrX, tltrY), (blbrX, blbrY)))
		microtubules_length.append(dA)


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


img = image.copy()
for i in range(len(tltrX_list)):
	#cv2.line(img, (int(tltrX_list[i]), int(tltrY_list[i])), (int(blbrX_list[i]), int(blbrY_list[i])),(255, 0, 255), 2)
	cv2.circle(img, (int(endpoints_list[i][0][0]), int(endpoints_list[i][0][1])),2, (0, 255, 255), -1)
	cv2.circle(img, (int(endpoints_list[i][1][0]), int(endpoints_list[i][1][1])),2, (0, 255, 255), -1)
	#cv2.putText(img, "{:.1f}".format(dA_list[i]), (int(tltrX_list[i] - 15), int(tltrY_list[i] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
	#cv2.putText(img, "{:d}".format(i), (int(endpoints_list[i][0][0] + 10), int(endpoints_list[i][0][1])), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 255, 255), 2)
#cv2.imshow("Image", img)

cv2.imwrite("Region_Segmentation/region_growing_result.png", img)

cv2.imshow('Region Growing', img)
cv2.waitKey(0)