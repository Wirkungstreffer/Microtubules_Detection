import glob
import cv2
from cv2 import drawContours, rectangle
import os
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.defchararray import center
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


# Acquire microtubles position and calculate length information 
#########################################################################################################################

# Set a counter number of frame
frame = 1

# Create length and width recorder for each seed in each frame
seed_correspond_microtubules_length = []
seed_correspond_microtubules_width = []

# Set up deviation pixels tolerance
tolerance = tolerance_pixels

# Create list to store the number of length
Length_Micotubulues = []

# Visualize measurements images list
visualize_measurements_images_list = []

prediction_measurements_images_list = []

overlapping_images_list = []

# Crop image list
crop_images_list = []

# Measure the lengths of microtubules segmentation
for image in array_of_predict_input_image:

    # Connect separate parts
    dilataion = cv2.dilate(image, None, iterations=1)
    erosion = cv2.erode(dilataion, None, iterations=1)

    # Read the original image
    input_original = array_of_input_image[frame-1]

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
        if areas[i] >= 5:   
            image_noise_reduce[labels == i + 1] = 255

    # Get contours of segmentations
    cnts = cv2.findContours(image_noise_reduce, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    pixelsPerMetric = None

    # Create lists to store the information of mininal rectangular
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

    microtubules_endpoints_list = []

    Length_Micotubulues_list = []

    image_copy = image.copy()
    # Loop for all contour
    for c in cnts:
        
        # if it's too small, it might be noise, just ignore it
        if cv2.contourArea(c) < 5:
            continue

        epsilon = 0.15*cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,epsilon,True)

        # If approximate points number is larger than 3, means there could be overlapping
        if (len(approx) >= 3) & (cv2.contourArea(c) > 200):

            # Create a list to store overlapping coordinate
            overlapping_coordinate_list = []

            for coordinate in approx:
                overlapping_coordinate_list.append(coordinate[0])

            # Make a internal pair number sublist, for example[(0,1),(1,2),(0,2)]
            stuff = list(np.arange(0,len(overlapping_coordinate_list)))
            subset_list = []
            for L in range(0, len(stuff)+1):
                for subset in itertools.combinations(stuff, L):
                    if len(subset) == 2:
                        subset_list.append(subset)
                        
            # Create a list to store calculated different pixels
            different_pixels_list = []

            # Calculate the pixels differents between line-added image and original image to verify which points couple are useful
            for sub in range(len(subset_list)):
                predict_image_orig = image_copy.copy()
                cv2.line(predict_image_orig , (tuple(overlapping_coordinate_list[subset_list[sub][0]])), tuple((overlapping_coordinate_list[subset_list[sub][1]])),(255, 255, 255), 3)
                sub_img = cv2.subtract(predict_image_orig, image_copy)
                #sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
                non_zero_pixel = cv2.countNonZero(sub_img)
                different_pixels_list.append(non_zero_pixel)

            # According to approximation number to know the overlapped microtubules quantity
            pair_number = round(len(approx)/2)

            # Select the pair points that have minimal different
            small_number = heapq.nsmallest(pair_number, different_pixels_list) 
            small_index = []
            for t in small_number:
                index = different_pixels_list.index(t)
                small_index.append(index)
                different_pixels_list[index] = 0
            

            # Add the correct points into the width points lists
            for min_index in small_index:
                tltrX = overlapping_coordinate_list[subset_list[min_index][0]][0]
                tltrX_list.append(tltrX)
                tltrY = overlapping_coordinate_list[subset_list[min_index][0]][1]
                tltrY_list.append(tltrY)
                tlblX = overlapping_coordinate_list[subset_list[min_index][0]][0]
                tlblX_list.append(tlblX)
                tlblY = overlapping_coordinate_list[subset_list[min_index][0]][1]
                tlblY_list.append(tlblY)
                blbrX = overlapping_coordinate_list[subset_list[min_index][1]][0]
                blbrX_list.append(blbrX)
                blbrY = overlapping_coordinate_list[subset_list[min_index][1]][1]
                blbrY_list.append(blbrY)
                trbrX = overlapping_coordinate_list[subset_list[min_index][1]][0]
                trbrX_list.append(trbrX)
                trbrY = overlapping_coordinate_list[subset_list[min_index][1]][1]
                trbrY_list.append(trbrY)
                
                # Calculate the length and width distances
                dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                
                # Add the distance information to the lists
                dA_list.append(dA)
                dB_list.append(dB)

                if dB >= dA:
                    microtubules_endpoints_list.append([(tlblX, tlblY),(trbrX, trbrY)])
                    Length_Micotubulues_list.append(dB)
                else:
                    microtubules_endpoints_list.append([(tltrX, tltrY),(blbrX, blbrY)])
                    Length_Micotubulues_list.append(dA)
                    
                      
        # If approximate points number is smaller than 3, means no overlapping
        else:
            
            # Use minimal area rectangular to box the segmentation
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)

            boxes.append(box.astype("int"))

            # Get the midpoint of the length and width of the box
            (tl, tr, br, bl) = box
            
            # Midpoints of the width sides of box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            
            # Midpoints of the length sides of box
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # Add the points into the lists
            tltrX_list.append(tltrX)
            tltrY_list.append(tltrY)
            blbrX_list.append(blbrX)
            blbrY_list.append(blbrY)

            tlblX_list.append(tlblX)
            tlblY_list.append(tlblY)
            trbrX_list.append(trbrX)
            trbrY_list.append(trbrY)

            # Calculate the length and width distances
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            # Add the distance information to the lists
            dA_list.append(dA)
            dB_list.append(dB)

            # Add the endpoints coordinates into list
            if dB >= dA:
                microtubules_endpoints_list.append([(tlblX, tlblY),(trbrX, trbrY)])
                Length_Micotubulues_list.append(dB)
            else:
                microtubules_endpoints_list.append([(tltrX, tltrY),(blbrX, blbrY)])
                Length_Micotubulues_list.append(dA)

    # Convert the list to array for the further process
    tltrX_list = np.array(tltrX_list)
    tltrY_list = np.array(tltrY_list)
    blbrX_list = np.array(blbrX_list)
    blbrY_list = np.array(blbrY_list)

    tlblX_list = np.array(tlblX_list)
    tlblY_list = np.array(tlblY_list)
    trbrX_list = np.array(trbrX_list)
    trbrY_list = np.array(trbrY_list)

    dA_list = np.array(dA_list)
    dB_list = np.array(dB_list)

    Length_Micotubulues_list = np.array(Length_Micotubulues_list)
    
    # The lengths of microtubules that detected and measured in video frames
    Length_Micotubulues.append(Length_Micotubulues_list)

    
    # The lengths of microtubules which concatenation with corresponding seed
    for seed_endpoint in seed_endpoints_list:
        
        # Store two endpoints of seeds
        seed_endpoint_1 = seed_endpoint[0]
        seed_endpoint_2 = seed_endpoint[1]
        
        # Create lists for store the index of the microtubules corresponding to seeds
        correspond_mt = []
        correspond_mt_per_seed = []
        min_index_list = []
        min_val_list = []
        seed_endpoint_validation_number = 0
        
        # Calculate the distance between each seeds and microtubules, save the microtubules have smaller distance than tolerance with seeds
        for mt in range(len(microtubules_endpoints_list)):
            
            # Caculate the distance between seeds endpoints and microtubules endpoints
            distance_1 = dist.euclidean(seed_endpoint_1, (microtubules_endpoints_list[mt][0][0], microtubules_endpoints_list[mt][0][1]))
            distance_2 = dist.euclidean(seed_endpoint_1, (microtubules_endpoints_list[mt][1][0], microtubules_endpoints_list[mt][1][1]))
            distance_3 = dist.euclidean(seed_endpoint_2, (microtubules_endpoints_list[mt][0][0], microtubules_endpoints_list[mt][0][1]))
            distance_4 = dist.euclidean(seed_endpoint_2, (microtubules_endpoints_list[mt][1][0], microtubules_endpoints_list[mt][1][1]))

            # Save the microtubules index which has smaller distance than tolerance
            if distance_1 < tolerance:
                correspond_mt.append(mt)
                seed_endpoint_validation_number = 1 
            elif distance_2 < tolerance:
                correspond_mt.append(mt)
                seed_endpoint_validation_number = 1  
            elif distance_3 < tolerance:
                correspond_mt.append(mt) 
                seed_endpoint_validation_number = 2
            elif distance_4 < tolerance:
                correspond_mt.append(mt) 
                seed_endpoint_validation_number = 2
            else:
                correspond_mt.append(len(microtubules_endpoints_list) + 200)

        # Make the stored list into sublist according to each seed
        for mt_number in range(0,len(correspond_mt),len(microtubules_endpoints_list)):
            number_group = correspond_mt[mt_number:mt_number+len(microtubules_endpoints_list)]
            correspond_mt_per_seed.append(number_group)

        # Find the microtubules corresponding to the seed, sometimes there could be two microtubules in both seeds endpoints
        correspond_indexes = []
        for index_number in range(len(correspond_mt_per_seed[0])):
            if correspond_mt_per_seed[0][index_number] < (len(microtubules_endpoints_list) + 200):
                correspond_indexes.append(index_number)

        # Only need the longer microtubules, delete the shorter microtubules information
        if len(correspond_indexes) == 2:
            if dB_list[correspond_indexes[0]] > dB_list[correspond_indexes[1]]:
                correspond_mt_per_seed[0][correspond_indexes[1]] = len(microtubules_endpoints_list) + 200
            elif dB_list[correspond_indexes[0]] < dB_list[correspond_indexes[1]]:
                correspond_mt_per_seed[0][correspond_indexes[0]] = len(microtubules_endpoints_list) + 200
        
        # Find the corresponding index for each seed
        for the_index in correspond_mt_per_seed:
            min_index_list.append(correspond_mt.index(min(the_index)))
            min_val_list.append(min(the_index))
        
        # Save the seed corresponding microtubules length information to the seed_correspond_microtubules lists
        for x in range(len(min_index_list)):
            if min_val_list[x] == len(microtubules_endpoints_list) + 200 :
                seed_correspond_microtubules_length.append(-1)
            elif seed_endpoint_validation_number == 1 :
                #seed_correspond_microtubules_length.append(dB_list[min_index_list[x]])   dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                seed_correspond_microtubules_length.append(max(dist.euclidean(seed_endpoint_1,(microtubules_endpoints_list[min_index_list[x]][0][0],microtubules_endpoints_list[min_index_list[x]][0][1])), dist.euclidean(seed_endpoint_1,(microtubules_endpoints_list[min_index_list[x]][1][0],microtubules_endpoints_list[min_index_list[x]][1][1]))))
            elif seed_endpoint_validation_number == 2 :
                #seed_correspond_microtubules_length.append(dB_list[min_index_list[x]])   dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                seed_correspond_microtubules_length.append(max(dist.euclidean(seed_endpoint_2,(microtubules_endpoints_list[min_index_list[x]][0][0],microtubules_endpoints_list[min_index_list[x]][0][1])), dist.euclidean(seed_endpoint_1,(microtubules_endpoints_list[min_index_list[x]][1][0],microtubules_endpoints_list[min_index_list[x]][1][1]))))

    # Add seed image and microtubules image
    add_img = cv2.add(seed_original,input_original)

    # Draw the length & width line and the number
    orignal_composite = add_img.copy()
    crop_pre_image = input_original.copy()
    for i in range(len(microtubules_endpoints_list)):
        # Draw lines and informations of microtubules
        #cv2.line(orignal_composite, (int(tltrX_list[i]), int(tltrY_list[i])), (int(blbrX_list[i]), int(blbrY_list[i])),(255, 0, 255), 2)
        #cv2.line(orignal_composite, (int(tlblX_list[i]), int(tlblY_list[i])), (int(trbrX_list[i]), int(trbrY_list[i])),(255, 0, 255), 2)
        #cv2.putText(orignal_composite, "{:.1f}".format(dA_list[i]), (int(tltrX_list[i] - 15), int(tltrY_list[i] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
        #cv2.putText(orignal_composite, "{:.1f}".format(dB_list[i]), (int(trbrX_list[i] + 10), int(trbrY_list[i])), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 0, 255), 2)
        #cv2.circle(orignal_composite, (int(tltrX_list[i]), int(tltrY_list[i])), 2, (255, 0, 255), -1)
        #cv2.circle(orignal_composite, (int(blbrX_list[i]), int(blbrY_list[i])), 2, (255, 0, 255), -1)
        cv2.circle(orignal_composite, (int(microtubules_endpoints_list[i][0][0]), int(microtubules_endpoints_list[i][0][1])), 2, (255, 0, 255), -1)
        cv2.circle(orignal_composite, (int(microtubules_endpoints_list[i][1][0]), int(microtubules_endpoints_list[i][1][1])), 2, (255, 0, 255), -1)
        cv2.circle(crop_pre_image, (int(microtubules_endpoints_list[i][0][0]), int(microtubules_endpoints_list[i][0][1])), 2, (255, 0, 255), -1)
        cv2.circle(crop_pre_image, (int(microtubules_endpoints_list[i][1][0]), int(microtubules_endpoints_list[i][1][1])), 2, (255, 0, 255), -1)
    
    crop_images_list.append(crop_pre_image)

    for j in range(len(seed_endpoints_list)):
        # Draw lines and informations of microtubules
        #cv2.line(orignal_composite, (int(seed_tltrX_list[j]), int(seed_tltrY_list[j])), (int(seed_blbrX_list[j]), int(seed_blbrY_list[j])),(0, 255, 255), 2)
        #cv2.line(orignal_composite, (int(seed_tlblX_list[j]), int(seed_tlblY_list[j])), (int(seed_trbrX_list[j]), int(seed_trbrY_list[j])),(0, 255, 255), 2)
        cv2.putText(orignal_composite, "{:d}".format(j+1), (int(seed_endpoints_list[j][0][0] - 15), int(seed_endpoints_list[j][0][1] - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 255), 2)
        #cv2.putText(orignal_composite, "{:.1f}".format(seed_dB_list[j]), (int(seed_trbrX_list[j] + 10), int(seed_trbrY_list[j])), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 255, 255), 2)
        #cv2.circle(orignal_composite, (int(seed_tltrX_list[j]), int(seed_tltrY_list[j])), 2, (0, 255, 255), -1)
        #cv2.circle(orignal_composite, (int(seed_blbrX_list[j]), int(seed_blbrY_list[j])), 2, (0, 255, 255), -1)
        cv2.circle(orignal_composite, (int(seed_endpoints_list[j][0][0]), int(seed_endpoints_list[j][0][1])), 2, (0, 255, 255), -1)
        cv2.circle(orignal_composite, (int(seed_endpoints_list[j][1][0]), int(seed_endpoints_list[j][1][1])), 2, (0, 255, 255), -1)

    # Add the visualized measurement to the list
    visualize_measurements_images_list.append(orignal_composite)

    # Add seed image and microtubules prediction image
    predict_add_img = cv2.add(predict_seed_image[0],image)

    # Draw the length & width line and the number
    predict_composite = image
    for k in range(len(microtubules_endpoints_list)):
        # Draw lines and informations of microtubules
        #cv2.line(orignal_composite, (int(tltrX_list[i]), int(tltrY_list[i])), (int(blbrX_list[i]), int(blbrY_list[i])),(255, 0, 255), 2)
        #cv2.line(orignal_composite, (int(tlblX_list[i]), int(tlblY_list[i])), (int(trbrX_list[i]), int(trbrY_list[i])),(255, 0, 255), 2)
        #cv2.putText(orignal_composite, "{:.1f}".format(dA_list[i]), (int(tltrX_list[i] - 15), int(tltrY_list[i] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
        #cv2.putText(orignal_composite, "{:.1f}".format(dB_list[i]), (int(trbrX_list[i] + 10), int(trbrY_list[i])), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 0, 255), 2)
        #cv2.circle(predict_composite, (int(tltrX_list[k]), int(tltrY_list[k])), 2, (255, 255, 255), -1)
        #cv2.circle(predict_composite, (int(blbrX_list[k]), int(blbrY_list[k])), 2, (255, 255, 255), -1)
        cv2.circle(predict_composite, (int(microtubules_endpoints_list[k][0][0]), int(microtubules_endpoints_list[k][0][1])), 2, (255, 255, 255), -1)
        cv2.circle(predict_composite, (int(microtubules_endpoints_list[k][1][0]), int(microtubules_endpoints_list[k][1][1])), 2, (255, 255, 255), -1)

    for l in range(len(seed_endpoints_list)):
        # Draw lines and informations of microtubules
        #cv2.line(orignal_composite, (int(seed_tltrX_list[l]), int(seed_tltrY_list[l])), (int(seed_blbrX_list[l]), int(seed_blbrY_list[l])),(0, 255, 255), 2)
        #cv2.line(orignal_composite, (int(seed_tlblX_list[l]), int(seed_tlblY_list[l])), (int(seed_trbrX_list[l]), int(seed_trbrY_list[l])),(0, 255, 255), 2)
        cv2.putText(predict_composite, "{:d}".format(l+1), (int(seed_endpoints_list[l][0][0] - 15), int(seed_endpoints_list[l][0][1] - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 2)
        #cv2.putText(orignal_composite, "{:.1f}".format(seed_dB_list[j]), (int(seed_trbrX_list[j] + 10), int(seed_trbrY_list[j])), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 255, 255), 2)
        #cv2.circle(orignal_composite, (int(seed_tltrX_list[j]), int(seed_tltrY_list[j])), 2, (0, 255, 255), -1)
        #cv2.circle(orignal_composite, (int(seed_blbrX_list[j]), int(seed_blbrY_list[j])), 2, (0, 255, 255), -1)
        cv2.circle(predict_composite, (int(seed_endpoints_list[l][0][0]), int(seed_endpoints_list[l][0][1])), 2, (0, 255, 255), -1)
        cv2.circle(predict_composite, (int(seed_endpoints_list[l][1][0]), int(seed_endpoints_list[l][1][1])), 2, (0, 255, 255), -1)

    # Add the visualized measurement to the list
    prediction_measurements_images_list.append(predict_composite)

    frame = frame + 1

    #cv2.imshow("Measurement Visualization", predict_composite)
    #cv2.waitKey(0)


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

def startpoint(endpoint, angle, hypotenuse, scale):
    p = list(endpoint)
    q = list(endpoint)
    
    #angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    #hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] + scale * hypotenuse * cos(angle)
    q[1] = p[1] + scale * hypotenuse * sin(angle)
    #cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    
    return q

def centroid(img):
    moments = cv2.moments(img, False)

    cX = float(moments["m10"] / moments["m00"])
    cY = float(moments["m01"] / moments["m00"])

    return [cX, cY]

def cropimage(image, center, theta, width, height):

   ''' 
   Rotates OpenCV image around center with angle theta (in deg)
   then crops the image according to width and height.
   '''

   # Uncomment for theta in radians
   #theta *= 180/np.pi

   shape = ( image.shape[1], image.shape[0] ) # cv2.warpAffine expects shape in (length, height)

   matrix = cv2.getRotationMatrix2D( center=center, angle=theta, scale=1 )
   image = cv2.warpAffine( src=image, M=matrix, dsize=shape )

   x = int( center[0] - width/2  )
   y = int( center[1] - height/2 )

   image = image[ y:y+height, x:x+width ]

   return image


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

seed_angles_list = []

kymograph_pre_cropped_images_list = []
inverse_kymograph_pre_cropped_images_list = []

# PCA for every contour in seeds image
for seed_c_index in range(len(seed_cnts)):
    
    # Draw each contour only for visualisation purposes
    #cv2.drawContours(seed_image_copy, seed_cnts, seed_i, (0, 0, 255), 1)

    # Find the orientation of each shape
    for indexs in non_nan_columns_index:
        if indexs == seed_c_index:
            # Read columns
            columns_data = data_length_csv[data_length_csv.columns[indexs]]
            max_length = max(columns_data)

            seed_angle = getOrientation(seed_cnts[seed_c_index])
            ju_w_1, ju_w_2, ju_h_1, ju_h_2 = cropbox(seed_endpoints_list[seed_c_index][0], seed_angle, max(seed_length_list)+10, 10, 1)
            upper_bound_x = max(ju_w_1[0], ju_w_2[0], ju_h_1[0], ju_h_2[0])
            lower_bound_x = min(ju_w_1[0], ju_w_2[0], ju_h_1[0], ju_h_2[0])
            upper_bound_y = max(ju_w_1[1], ju_w_2[1], ju_h_1[1], ju_h_2[1])
            lower_bound_y = min(ju_w_1[1], ju_w_2[1], ju_h_1[1], ju_h_2[1])
            if (lower_bound_x <= seed_endpoints_list[seed_c_index][1][0] <= upper_bound_x) & (lower_bound_y <= seed_endpoints_list[seed_c_index][1][1] <= upper_bound_y):
                seed_angle = seed_angle + pi
            
            inverse_seed_angle = seed_angle + pi

            start_point = startpoint(seed_endpoints_list[seed_c_index][0], seed_angle, 10, 1)
            inverse_start_point = startpoint(seed_endpoints_list[seed_c_index][1], seed_angle, 10, 1)

            w_1, w_2, h_1, h_2 = cropbox(start_point, seed_angle, max_length+20, 5, 1)
            cv2.line(seed_image_copy, (int(start_point[0]),int(start_point[1])), (int(w_1[0]), int(w_1[1])), (0,0,225), 1, cv2.LINE_AA)
            cv2.line(seed_image_copy, (int(start_point[0]),int(start_point[1])), (int(w_2[0]), int(w_2[1])), (0,0,225), 1, cv2.LINE_AA)
            cv2.line(seed_image_copy, (int(w_1[0]), int(w_1[1])), (int(h_1[0]), int(h_1[1])), (0,0,225), 1, cv2.LINE_AA)
            cv2.line(seed_image_copy, (int(w_2[0]), int(w_2[1])), (int(h_2[0]), int(h_2[1])), (0,0,225), 1, cv2.LINE_AA)
            cv2.line(seed_image_copy, (int(h_1[0]), int(h_1[1])), (int(h_2[0]), int(h_2[1])), (0,0,225), 1, cv2.LINE_AA)

            inverse_w_1, inverse_w_2, inverse_h_1, inverse_h_2 = cropbox(inverse_start_point, inverse_seed_angle, max_length+20, 5, 1)
            cv2.line(seed_image_copy, (int(inverse_start_point[0]),int(start_point[1])), (int(inverse_w_1[0]), int(inverse_w_1[1])), (0,0,225), 1, cv2.LINE_AA)
            cv2.line(seed_image_copy, (int(inverse_start_point[0]),int(start_point[1])), (int(inverse_w_2[0]), int(inverse_w_2[1])), (0,0,225), 1, cv2.LINE_AA)
            cv2.line(seed_image_copy, (int(inverse_w_1[0]), int(inverse_w_1[1])), (int(inverse_h_1[0]), int(inverse_h_1[1])), (0,0,225), 1, cv2.LINE_AA)
            cv2.line(seed_image_copy, (int(inverse_w_2[0]), int(inverse_w_2[1])), (int(inverse_h_2[0]), int(inverse_h_2[1])), (0,0,225), 1, cv2.LINE_AA)
            cv2.line(seed_image_copy, (int(inverse_h_1[0]), int(inverse_h_1[1])), (int(inverse_h_2[0]), int(inverse_h_2[1])), (0,0,225), 1, cv2.LINE_AA)


            seed_angles_list.append(seed_angle)

            for image in crop_images_list:
                drawbox_image = image.copy()
                center_point = (int((w_1[0] + h_2[0])/2),int((w_1[1] + h_2[1])/2))
                cv2.line(drawbox_image, (int(start_point[0]),int(start_point[1])), (int(w_1[0]), int(w_1[1])), (0,0,225), 1, cv2.LINE_AA)
                cv2.line(drawbox_image, (int(start_point[0]),int(start_point[1])), (int(w_2[0]), int(w_2[1])), (0,0,225), 1, cv2.LINE_AA)
                cv2.line(drawbox_image, (int(w_1[0]), int(w_1[1])), (int(h_1[0]), int(h_1[1])), (0,0,225), 1, cv2.LINE_AA)
                cv2.line(drawbox_image, (int(w_2[0]), int(w_2[1])), (int(h_2[0]), int(h_2[1])), (0,0,225), 1, cv2.LINE_AA)
                cv2.line(drawbox_image, (int(h_1[0]), int(h_1[1])), (int(h_2[0]), int(h_2[1])), (0,0,225), 1, cv2.LINE_AA)
                cv2.circle(drawbox_image, center_point, 2, (0,255,0), -1)

                inverse_center_point = (int((inverse_w_1[0] + inverse_h_2[0])/2),int((inverse_w_1[1] + inverse_h_2[1])/2))
                cv2.line(drawbox_image, (int(inverse_start_point[0]),int(inverse_start_point[1])), (int(inverse_w_1[0]), int(inverse_w_1[1])), (0,0,225), 1, cv2.LINE_AA)
                cv2.line(drawbox_image, (int(inverse_start_point[0]),int(inverse_start_point[1])), (int(inverse_w_2[0]), int(inverse_w_2[1])), (0,0,225), 1, cv2.LINE_AA)
                cv2.line(drawbox_image, (int(inverse_w_1[0]), int(inverse_w_1[1])), (int(inverse_h_1[0]), int(inverse_h_1[1])), (0,0,225), 1, cv2.LINE_AA)
                cv2.line(drawbox_image, (int(inverse_w_2[0]), int(inverse_w_2[1])), (int(inverse_h_2[0]), int(inverse_h_2[1])), (0,0,225), 1, cv2.LINE_AA)
                cv2.line(drawbox_image, (int(inverse_h_1[0]), int(inverse_h_1[1])), (int(inverse_h_2[0]), int(inverse_h_2[1])), (0,0,225), 1, cv2.LINE_AA)
                cv2.circle(drawbox_image, inverse_center_point, 2, (0,255,0), -1)

                
                crop_box_image_copy = image.copy()
                inverse_crop_box_image_copy = image.copy()
                crop_images_case = cropimage(crop_box_image_copy, center_point, np.rad2deg(seed_angle % (2 * np.pi))-90, 10, int(max_length)+20)
                inverse_crop_images_case = cropimage(crop_box_image_copy, inverse_center_point, np.rad2deg(inverse_seed_angle % (2 * np.pi))-90, 10, int(max_length)+20)
                kymograph_pre_cropped_images_list.append(crop_images_case)
                inverse_kymograph_pre_cropped_images_list.append(inverse_crop_images_case)

#cv2.imshow("seed_pca",seed_image_copy)
#cv2.waitKey(0)

# Each sequence has 12 images
image_quantity_each_sequence = frame -1

# Create image sequence list
kymograph_images_sequence = []
inverse_kymograph_images_sequence = []

# Read each 12 images, save them as sublist of image sequence list
for img in range(0,len(kymograph_pre_cropped_images_list),image_quantity_each_sequence):
  image_group = kymograph_pre_cropped_images_list[img:img+image_quantity_each_sequence]
  inverse_image_group = inverse_kymograph_pre_cropped_images_list[img:img+image_quantity_each_sequence]
  kymograph_images_sequence.append(image_group)
  inverse_kymograph_images_sequence.append(inverse_image_group)


kymograph_list = []
inverse_kymograph_list = []
for kymo in range(len(kymograph_images_sequence)):

    kymograph = np.concatenate((kymograph_images_sequence[kymo]), axis=1)
    inverse_kymograph = np.concatenate((inverse_kymograph_images_sequence[kymo]), axis=1)

    cv2.imwrite("Semantic_Segmentation/implementation/data_output/Seed_%d_Corresponding_Kymograph.png" % (non_nan_columns_index[kymo]+1), kymograph)
    cv2.imwrite("Semantic_Segmentation/implementation/data_output/Seed_%d_Corresponding_Inverse_Kymograph.png" % (non_nan_columns_index[kymo]+1), inverse_kymograph)

    kymograph_list.append(kymograph)
    inverse_kymograph_list.append(inverse_kymograph)



