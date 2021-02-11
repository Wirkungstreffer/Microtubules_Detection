import cv2
import numpy as np

tiff_image = cv2.imread("200818_xb_reaction2_6um002_seedsc1t1.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
image = cv2.normalize(tiff_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#cv2.imshow("Image", image)

#gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# keypoints
sift = cv2.xfeatures2d.SIFT_create()

kp = sift.detect(image,None)

img=cv2.drawKeypoints(image,kp,outImage=None)

cv2.imshow('sift_keypoints.jpg',img)

cv2.waitKey(0)