from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from PIL import Image
import numpy as np
import argparse
import imutils
import cv2

# center point
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# parameters
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True, help="path to the input image")
#ap.add_argument("-w", "--width", type=float, required=True, help="width of the left-most object in the image (in inches)")
#args = vars(ap.parse_args())

# read imgae
image = cv2.imread("1.png")
cv2.imshow("image", image)
imgsize = image.shape
print(imgsize)

# gray grade
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# bilateral filter
bilateral = cv2.bilateralFilter(gray,3,1,1)

# median filter
median = cv2.medianBlur(gray,3)

# gauss filter to denoise
gauss = cv2.GaussianBlur(gray, (3, 3), 0)

# edge detection
edged = cv2.Canny(median, 50, 100)
cv2.imshow("edge", edged)

# fill up the gap within objects
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# find objects
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# order the sequence from left to right of the object
(cnts, _) = contours.sort_contours(cnts)

pixelsPerMetric = None

# for loop for all contour
for c in cnts:
	# if it's too samll, it might be noise, just ignore it
	if cv2.contourArea(c) < 40:
		continue

	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	
	box = perspective.order_points(box)
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)

	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)

	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)

	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	#if pixelsPerMetric is None:
	#	pixelsPerMetric = dB / imgsize[1]

	#dimA = dA / pixelsPerMetric
	#dimB = dB / pixelsPerMetric
 
	cv2.putText(orig, "{:.1f}".format(dA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
	cv2.putText(orig, "{:.1f}".format(dB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
 
	cv2.imshow("Image", orig)
	cv2.waitKey(0)