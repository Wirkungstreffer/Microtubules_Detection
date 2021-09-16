import glob
import cv2
from cv2 import drawContours
import os
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.type_check import imag
from scipy.spatial import distance as dist
from imutils import perspective
import imutils
import csv
from skimage import io
import itertools
import heapq
from sklearn.decomposition import PCA
import math
from math import atan2, cos, sin, sqrt, pi
import pandas as pd
import numpy as np

# Change the tolerance pixels of seeds microtubules concatenation if adjustment is needed
tolerance_pixels = 8


# If cv2.imshow() function dosen't work, use the followed line of code
os.environ['DISPLAY'] = ':1'

# Define a images loading function
def load_images(image_file_directory,channel):
    # Create a image list
    images = []
    
    # Check if the input folder exist
    if os.path.exists(image_file_directory)==False:
        raise FileNotFoundError( 'No such file or directory:'+ image_file_directory)

    # Reading the images in the folder 
    for directory_path in glob.glob(image_file_directory):
        image_path = glob.glob(os.path.join(directory_path, '*.png'))
        
        # Make sure reading sequence of the images is correctly according to the name sequence of images
        image_path.sort()
        
         # Reading images
        for i in image_path:
            
            if channel == 3:
                # Read the images as RGB mode
                img = cv2.imread(i, cv2.IMREAD_COLOR)
            elif channel == 1:
                # Read the images as binary mode
                img = cv2.imread(i, 0)
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            else:
                print("False channel input")

            # Add up into images list     
            images.append(img)
    
    return images

# Get center point
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Set the images and predictions paths
seed_path = "Semantic_Segmentation/implementation/seed_image/"
predict_seed_path = "Semantic_Segmentation/implementation/prediction_seed/"
images_path = "Semantic_Segmentation/implementation/input_image/"
predict_image_path = "Semantic_Segmentation/implementation/prediction_image/"

# Load original images and predicted images
seed_image = load_images(seed_path,3)
predict_seed_image = load_images(predict_seed_path,1)
array_of_input_image = load_images(images_path, 3)
array_of_predict_input_image = load_images(predict_image_path, 1)

# Acquire seeds position and calculate length information 
#########################################################################################################################

# Connect separate parts
seed_dilataion = cv2.dilate(predict_seed_image[0], None, iterations=1)
seed_erosion = cv2.erode(seed_dilataion, None, iterations=1)

# Read the original image
seed_original = seed_image[0]
predict_seed_image_original = predict_seed_image[0]
# Convert prdiction image to binary map by thresholding
seed_ret, seed_binary_map = cv2.threshold(seed_erosion,127,255,0)

# Get contours of segmentations
seed_cnts = cv2.findContours(seed_binary_map , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
seed_cnts = imutils.grab_contours(seed_cnts)

pixelsPerMetric = None

# Create lists to store the information of mininal rectangular
seed_boxes = []
seed_tltrX_list = []
seed_tltrY_list = []
seed_blbrX_list = []
seed_blbrY_list = []

seed_tlblX_list = []
seed_tlblY_list = []
seed_trbrX_list = []
seed_trbrY_list = []

seed_dA_list = []
seed_dB_list = []

# Create list for seeds endpoints
seed_endpoints_list = []
seed_length_list = []

# for loop for all contour
for seed_c in seed_cnts:
    
    # Apply approxPolyDP to find the approximate discontinues point to detect overlapping 
    seed_epsilon = 0.15*cv2.arcLength(seed_c,True)
    seed_approx = cv2.approxPolyDP(seed_c,seed_epsilon,True)

    # If approximate points number is larger than 3, means there could be overlapping
    if (len(seed_approx) >= 3) & (cv2.contourArea(seed_c) > 200):

        # Create a list to store overlapping coordinate
        seed_overlapping_coordinate_list = []

        for seed_coordinate in seed_approx:
            seed_overlapping_coordinate_list.append(seed_coordinate[0])

        # Make a internal pair number sublist, for example[(0,1),(1,2),(0,2)]
        seed_stuff = list(np.arange(0,len(seed_overlapping_coordinate_list)))
        seed_subset_list = []
        for seed_L in range(0, len(seed_stuff)+1):
            for seed_subset in itertools.combinations(seed_stuff, seed_L):
                if len(seed_subset) == 2:
                    seed_subset_list.append(seed_subset)
                    
        # Create a list to store calculated different pixels
        seed_different_pixels_list = []

        # Calculate the pixels differents between line-added image and original image to verify which points couple are useful
        for seed_sub in range(len(seed_subset_list)):
            seed_img_orig = predict_seed_image_original.copy()
            cv2.line(seed_img_orig , (tuple(seed_overlapping_coordinate_list[seed_subset_list[seed_sub][0]])), tuple((seed_overlapping_coordinate_list[seed_subset_list[seed_sub][1]])),(255, 255, 255), 3)
            seed_sub_img = cv2.subtract(seed_img_orig, predict_seed_image_original)
            #seed_sub_img = cv2.cvtColor(seed_sub_img, cv2.COLOR_BGR2GRAY)
            seed_non_zero_pixel = cv2.countNonZero(seed_sub_img)
            seed_different_pixels_list.append(seed_non_zero_pixel)

        # According to approximation number to know the overlapped microtubules quantity
        seed_pair_number = round(len(seed_approx)/2)

        # Select the pair points that have minimal different
        seed_small_number = heapq.nsmallest(seed_pair_number, seed_different_pixels_list) 
        seed_small_index = []
        for seed_t in seed_small_number:
            seed_index = seed_different_pixels_list.index(seed_t)
            seed_small_index.append(seed_index)
            seed_different_pixels_list[seed_index] = 0
        

        # Add the correct points into the width points lists
        for seed_min_index_one in seed_small_index:
            seed_tltrX_list.append(seed_overlapping_coordinate_list[seed_subset_list[seed_min_index_one][0]][0])
            seed_tltrY_list.append(seed_overlapping_coordinate_list[seed_subset_list[seed_min_index_one][0]][1])
            seed_tlblX_list.append(seed_overlapping_coordinate_list[seed_subset_list[seed_min_index_one][0]][0])
            seed_tlblY_list.append(seed_overlapping_coordinate_list[seed_subset_list[seed_min_index_one][0]][1])
        
        # Add the correct points into the lengthpoints lists
        for seed_min_index_two in seed_small_index:
            seed_blbrX_list.append(seed_overlapping_coordinate_list[seed_subset_list[seed_min_index_two][1]][0])
            seed_blbrY_list.append(seed_overlapping_coordinate_list[seed_subset_list[seed_min_index_two][1]][1])
            seed_trbrX_list.append(seed_overlapping_coordinate_list[seed_subset_list[seed_min_index_two][1]][0])
            seed_trbrY_list.append(seed_overlapping_coordinate_list[seed_subset_list[seed_min_index_two][1]][1])
        
        # Add the correct points into the endpoints lists
        for seed_min_index_endpoint in seed_small_index:
            seed_endpoints_list.append([(seed_overlapping_coordinate_list[seed_subset_list[seed_min_index_endpoint][0]][0], seed_overlapping_coordinate_list[seed_subset_list[seed_min_index_endpoint][0]][1]),(seed_overlapping_coordinate_list[seed_subset_list[seed_min_index_endpoint][1]][0], seed_overlapping_coordinate_list[seed_subset_list[seed_min_index_endpoint][1]][1])])
    
    # If approximate points number is smaller than 3, means no overlapping
    else:

        # Use minimal area rectangular to box the segmentation
        seed_box = cv2.minAreaRect(seed_c)
        seed_box = cv2.cv.BoxPoints(seed_box) if imutils.is_cv2() else cv2.boxPoints(seed_box)
        seed_box = np.array(seed_box, dtype="int")
        seed_box = perspective.order_points(seed_box)

        seed_boxes.append(seed_box.astype("int"))

        # Get the midpoint of the length and width of the box
        (seed_tl, seed_tr, seed_br, seed_bl) = seed_box
        
        # Midpoints of the width sides of box
        (seed_tltrX, seed_tltrY) = midpoint(seed_tl, seed_tr)
        (seed_blbrX, seed_blbrY) = midpoint(seed_bl, seed_br)

        # Midpoints of the length sides of box
        (seed_tlblX, seed_tlblY) = midpoint(seed_tl, seed_bl)
        (seed_trbrX, seed_trbrY) = midpoint(seed_tr, seed_br)

        # Add the width sides midpoints into the lists
        seed_tltrX_list.append(seed_tltrX)
        seed_tltrY_list.append(seed_tltrY)
        seed_blbrX_list.append(seed_blbrX)
        seed_blbrY_list.append(seed_blbrY)

        # Add the length sides midpoints into the lists
        seed_tlblX_list.append(seed_tlblX)
        seed_tlblY_list.append(seed_tlblY)
        seed_trbrX_list.append(seed_trbrX)
        seed_trbrY_list.append(seed_trbrY)

        # Calculate the length and width distances
        seed_dA = dist.euclidean((seed_tltrX, seed_tltrY), (seed_blbrX, seed_blbrY))
        seed_dB = dist.euclidean((seed_tlblX, seed_tlblY), (seed_trbrX, seed_trbrY))

        # Add the distance information to the lists
        seed_dA_list.append(seed_dA)
        seed_dB_list.append(seed_dB)

        # Add the endpoints coordinates into list
        if seed_dB >= seed_dA:
            seed_endpoints_list.append([(seed_tlblX, seed_tlblY),(seed_trbrX, seed_trbrY)])
            seed_length_list.append(seed_dB)
        else:
            seed_endpoints_list.append([(seed_tltrX, seed_tltrY),(seed_blbrX, seed_blbrY)])
            seed_length_list.append(seed_dA)


def drawAxis(img, p_, q_, hypotenuse, scale, colour):
    p = list(p_)
    q = list(q_)
    w_1 = list(q_)
    w_2 = list(q_)
    
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    #hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    
    # Create rectangular
    vertical_angle_1 = pi/2 + angle
    w_1[0] = p[0] - scale * 50 * cos(vertical_angle_1)
    w_1[1] = p[1] - scale * 50 * sin(vertical_angle_1)
    cv2.line(img, (int(p[0]), int(p[1])), (int(w_1[0]), int(w_1[1])), (255, 0, 0), 1, cv2.LINE_AA)
    vertical_angle_2 = pi + vertical_angle_1
    w_2[0] = p[0] - scale * 50 * cos(vertical_angle_2)
    w_2[1] = p[1] - scale * 50 * sin(vertical_angle_2)
    cv2.line(img, (int(p[0]), int(p[1])), (int(w_2[0]), int(w_2[1])), (0, 255, 0), 1, cv2.LINE_AA)

    # Create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    #cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    #cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

def getOrientation(pts):
    # Transfer segmentations into data points
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    
    # Draw the circle
    #cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    
    # Draw the axis that represent eigenvector orientation
    #drawAxis(img, cntr, p1, 30, 1,(0, 0, 255))
    #drawAxis(img, cntr, p2, (255, 255, 0), 5)
    
    # Calculate angles
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    
    return angle

def drawcropbox(img, start_point, angle, hypotenuse, width, scale, colour):
    p = list(start_point)
    q = list(start_point)
    w_1 = list(start_point)
    w_2 = list(start_point)
    h_1 = list(start_point)
    h_2 = list(start_point)
    
    #angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    #hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    #cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    
    # Draw crop rectangular
    vertical_angle_1 = pi/2 + angle
    w_1[0] = p[0] - scale * width * cos(vertical_angle_1)
    w_1[1] = p[1] - scale * width * sin(vertical_angle_1)
    cv2.line(img, (int(p[0]), int(p[1])), (int(w_1[0]), int(w_1[1])), colour, 1, cv2.LINE_AA)
    vertical_angle_2 = pi + vertical_angle_1
    w_2[0] = p[0] - scale * width * cos(vertical_angle_2)
    w_2[1] = p[1] - scale * width * sin(vertical_angle_2)
    cv2.line(img, (int(p[0]), int(p[1])), (int(w_2[0]), int(w_2[1])), colour, 1, cv2.LINE_AA)

    h_1[0] = w_1[0] - scale * hypotenuse * cos(angle)
    h_1[1] = w_1[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(w_1[0]), int(w_1[1])), (int(h_1[0]), int(h_1[1])), colour, 1, cv2.LINE_AA)
    h_2[0] = w_2[0] - scale * hypotenuse * cos(angle)
    h_2[1] = w_2[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(w_2[0]), int(w_2[1])), (int(h_2[0]), int(h_2[1])), colour, 1, cv2.LINE_AA)

    cv2.line(img, (int(h_1[0]), int(h_1[1])), (int(h_2[0]), int(h_2[1])), colour, 1, cv2.LINE_AA)

def cropbox(start_point, angle, hypotenuse, width, scale):
    p = list(start_point)
    q = list(start_point)
    w_1 = list(start_point)
    w_2 = list(start_point)
    h_1 = list(start_point)
    h_2 = list(start_point)
    
    #angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    #hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    #cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    
    # Draw crop rectangular
    vertical_angle_1 = pi/2 + angle
    w_1[0] = p[0] - scale * width * cos(vertical_angle_1)
    w_1[1] = p[1] - scale * width * sin(vertical_angle_1)
    #cv2.line(img, (int(p[0]), int(p[1])), (int(w_1[0]), int(w_1[1])), colour, 1, cv2.LINE_AA)
    vertical_angle_2 = pi + vertical_angle_1
    w_2[0] = p[0] - scale * width * cos(vertical_angle_2)
    w_2[1] = p[1] - scale * width * sin(vertical_angle_2)
    #cv2.line(img, (int(p[0]), int(p[1])), (int(w_2[0]), int(w_2[1])), colour, 1, cv2.LINE_AA)

    h_1[0] = w_1[0] - scale * hypotenuse * cos(angle)
    h_1[1] = w_1[1] - scale * hypotenuse * sin(angle)
    #cv2.line(img, (int(w_1[0]), int(w_1[1])), (int(h_1[0]), int(h_1[1])), colour, 1, cv2.LINE_AA)
    h_2[0] = w_2[0] - scale * hypotenuse * cos(angle)
    h_2[1] = w_2[1] - scale * hypotenuse * sin(angle)
    #cv2.line(img, (int(w_2[0]), int(w_2[1])), (int(h_2[0]), int(h_2[1])), colour, 1, cv2.LINE_AA)

    #cv2.line(img, (int(h_1[0]), int(h_1[1])), (int(h_2[0]), int(h_2[1])), colour, 1, cv2.LINE_AA)
    return w_1, w_2, h_1, h_2


# Read the csv file
data_length_csv = pd.read_csv("Semantic_Segmentation/implementation/data_output/Microtubules_Lengths_with_Seed_Concatenation.csv", header=None)

# Define a function to select non-negative column
def select_non_nan_column(column):
    # Create a list to store selected column
    column_non_nan = []
    
    # Filter all the negative
    for non_nan in column:
        if non_nan != -1:
            column_non_nan.append(non_nan)

    return column_non_nan

# Define a function that eliminate outliers 
def reject_outliers(data):
    # Create a list to store filtered data
    data_filtered = []
    data_non_nan = []
    
    # Caculate mean and variance of the data
    for n_z in data:
        if n_z != -1:
            data_non_nan.append(n_z)

    u = np.mean(data_non_nan)
    s = np.std(data_non_nan)

    # Save the data within 2 standard deviation
    for d in data_non_nan:
        if (d>(u-2*s)) & (d<(u+2*s)):
            data_filtered.append(d)
    
    return data_filtered

# Create a list to store the non-zero column index
non_nan_columns_index = []

# Keep the non-negative columns
for column_loop in range(data_length_csv.shape[1]):
    # Filter out negative columns
    column_validation = select_non_nan_column(data_length_csv[data_length_csv.columns[column_loop]])
    
    # Filter out the falsh & miss detected column
    if len(column_validation) >= 0.2*data_length_csv.shape[0]:
        non_nan_columns_index.append(column_loop) 


print(non_nan_columns_index)

seed_image_copy = (seed_image[0]).copy()
first_predict_input_image = (array_of_predict_input_image[0]).copy()

seed_angles_list = []
# PCA for every contour in seeds image
for seed_c_index in range(len(seed_cnts)):
    
    # Draw each contour only for visualisation purposes
    #cv2.drawContours(seed_image_copy, seed_cnts, seed_i, (0, 0, 255), 1)
    
    # Find the orientation of each shape
    for indexs in non_nan_columns_index:
        if indexs == seed_c_index:
            seed_angle = getOrientation(seed_cnts[seed_c_index])
            ju_w_1, ju_w_2, ju_h_1, ju_h_2 = cropbox(seed_endpoints_list[seed_c_index][0], seed_angle, max(seed_length_list), 10, 1)
            upper_bound_x = max(ju_w_1[0], ju_w_2[0], ju_h_1[0], ju_h_2[0])
            lower_bound_x = min(ju_w_1[0], ju_w_2[0], ju_h_1[0], ju_h_2[0])
            upper_bound_y = max(ju_w_1[1], ju_w_2[1], ju_h_1[1], ju_h_2[1])
            lower_bound_y = min(ju_w_1[1], ju_w_2[1], ju_h_1[1], ju_h_2[1])
            if (lower_bound_x <= seed_endpoints_list[seed_c_index][1][0] <= upper_bound_x) & (lower_bound_y <= seed_endpoints_list[seed_c_index][1][1] <= upper_bound_y):
                seed_angle = seed_angle + pi
            
            w_1, w_2, h_1, h_2 = cropbox(seed_endpoints_list[seed_c_index][0], seed_angle, 50, 5, 1)
            cv2.line(seed_image_copy, (int(seed_endpoints_list[seed_c_index][0][0]),int(seed_endpoints_list[seed_c_index][0][1])), (int(w_1[0]), int(w_1[1])), (0,0,225), 1, cv2.LINE_AA)
            cv2.line(seed_image_copy, (int(seed_endpoints_list[seed_c_index][0][0]),int(seed_endpoints_list[seed_c_index][0][1])), (int(w_2[0]), int(w_2[1])), (0,0,225), 1, cv2.LINE_AA)
            cv2.line(seed_image_copy, (int(w_1[0]), int(w_1[1])), (int(h_1[0]), int(h_1[1])), (0,0,225), 1, cv2.LINE_AA)
            cv2.line(seed_image_copy, (int(w_2[0]), int(w_2[1])), (int(h_2[0]), int(h_2[1])), (0,0,225), 1, cv2.LINE_AA)
            cv2.line(seed_image_copy, (int(h_1[0]), int(h_1[1])), (int(h_2[0]), int(h_2[1])), (0,0,225), 1, cv2.LINE_AA)

            seed_angles_list.append(seed_angle)


cv2.imshow("seed_pca",seed_image_copy)
cv2.waitKey(0)