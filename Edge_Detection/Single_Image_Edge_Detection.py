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

# center point
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# read imgae
#tiff_image = cv2.imread("Edge_Detection/8bit_tiff_file/Composite_z002_c002.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
#image = cv2.normalize(tiff_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
image = cv2.imread("Edge_Detection/input_image/200818_Xb_Reaction2_6uM003_016.png", cv2.IMREAD_COLOR)
imgsize = image.shape
#print(imgsize)
image_D_E = cv2.dilate(image, None, iterations=1) - cv2.erode(image, None, iterations=1) 
#cv2.imshow("image_D_E", image_D_E)
# gray grade
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# bilateral filter
bilateral = cv2.bilateralFilter(image,3,1,1)

# median filter
median = cv2.medianBlur(image,3)
#cv2.imshow("median", median)

# gauss filter to denoise
gauss = cv2.GaussianBlur(image, (5, 5), 0)
# NLM
#nlm = cv2.fastNlMeansDenoising(image, None, 10 ,10,7,21)

# edge detection
#median = np.uint8(median)
edged = cv2.Canny(gauss, 20, 100)	
#edged = filters.sobel(gauss)
#cv2.imshow("edge", edged)
#cv2.waitKey(0)

# fill up the gap within objects
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

#cv2.imshow("edge_filled", edged)

# find objects
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

	if dA < dB:
		endpoints_list.append(((tlblX, tlblY), (trbrX, trbrY)))
		microtubules_length.append(dB)
	else:
		endpoints_list.append(((tltrX, tltrY), (blbrX, blbrY)))
		microtubules_length.append(dA)
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


img = image.copy()
for i in range(len(tltrX_list)):
	#cv2.line(img, (int(tltrX_list[i]), int(tltrY_list[i])), (int(blbrX_list[i]), int(blbrY_list[i])),(255, 0, 255), 2)
	cv2.circle(img, (int(endpoints_list[i][0][0]), int(endpoints_list[i][0][1])),2, (0, 255, 255), -1)
	cv2.circle(img, (int(endpoints_list[i][1][0]), int(endpoints_list[i][1][1])),2, (0, 255, 255), -1)
	#cv2.putText(img, "{:.1f}".format(dA_list[i]), (int(tltrX_list[i] - 15), int(tltrY_list[i] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
	cv2.putText(img, "{:d}".format(i), (int(endpoints_list[i][0][0] + 10), int(endpoints_list[i][0][1])), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 255, 255), 2)
#cv2.imshow("Image", img)

img_box = image.copy()
for k in range(len(boxes)):
    cv2.drawContours(img_box, [boxes[k].astype("int")], -1, (0, 255, 0), 2)
#cv2.imshow("Image_box", img_box)

orig = image.copy()
cv2.imshow("predicted_image", img)
cv2.waitKey(0)

cv2.imwrite("Edge_Detection/Edge_detection_predicted_image.png", img)