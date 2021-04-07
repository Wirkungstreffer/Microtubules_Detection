import os
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
    for filename in os.listdir(r"./"+ directory):
        tiff_image = cv2.imread(directory + "/" + filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        img = cv2.normalize(tiff_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        cv2.imwrite("Microtubules_Tiff_Date" + "/" + filename, img)
        
        array_of_img.append(img)

read_directory("ND2")