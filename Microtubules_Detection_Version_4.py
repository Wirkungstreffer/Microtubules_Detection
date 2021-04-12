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
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import pwlf

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
        #print(array_of_img)

read_directory("Tiff_files")

# center point
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

Length_Micotubulues = []

for image in array_of_img:
    # gray grade
	#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	
	
	# bilateral filter
	bilateral = cv2.bilateralFilter(image,3,1,1)

	# median filter
	median = cv2.medianBlur(image,3)
	#cv2.imshow("median", median)

	# gauss filter to denoise
	gauss = cv2.GaussianBlur(image, (5, 5), 0)

	# edge detection
	#median = np.uint8(median)
	edged = cv2.Canny(gauss, 30, 120)
	#cv2.imshow("edge", edged)

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

	#creating date list for each image
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
    
	#caculate the length
	for c in cnts:
    	# if it's too samll, it might be noise, just ignore it
		if cv2.contourArea(c) < 30:
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

		dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
		dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

		dA_list.append(dA)
		dB_list.append(dB)
	
	#transfer data type from list to array
	tltrX_list = np.array(tltrX_list)
	tltrY_list = np.array(tltrY_list)
	blbrX_list = np.array(blbrX_list)
	blbrY_list = np.array(blbrY_list)

	tlblX_list = np.array(tlblX_list)
	tlblY_list = np.array(tlblY_list)
	trbrX_list = np.array(trbrX_list)
	trbrY_list = np.array(trbrY_list)
	
	#the list of length of each Microtubule in each image
	#print(len(dB_list))

	Length_Micotubulues.append(dB_list)

	img = image.copy()
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

	cv2.waitKey(0)

#test of the last one micotubulues for every image
Case_Microtubules = []
for m in Length_Micotubulues:
    Case_Microtubules.append(m[0])

#scatter plot
x = np.array([np.arange(0,len(Length_Micotubulues))])
y = np.array([Case_Microtubules])

plt.scatter(x, y)
plt.show()

#linear regression
x_l = np.array([np.arange(0,len(Length_Micotubulues))])
y_l = Case_Microtubules

my_pwlf = pwlf.PiecewiseLinFit(x_l, y_l)
breaks = my_pwlf.fit(3)
print(breaks)

x_hat = np.linspace(x.min(), x.max(), 100)
y_hat = my_pwlf.predict(x_hat)

plt.figure()
plt.plot(x, y, 'o')
plt.plot(x_hat, y_hat, '-')
plt.show()