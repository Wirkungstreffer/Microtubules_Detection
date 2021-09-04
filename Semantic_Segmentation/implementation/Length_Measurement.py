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


#### This script is to measure the lengths of microtubules and return the csv file recording the lengths information ####
#### The input is images in 4 folder "implementation/input_image", "implementation/prediction_image", "implementation/prediction_seed" and "implementation/seed_image" ####
#### Output of lengths information will be stored as csv file and some additional images to visualize the measurements ####


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


# Set data output path
output_path = "Semantic_Segmentation/implementation/data_output/"

if os.path.exists(output_path)==False:
    os.makedirs(output_path)

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

# Get the informations of segmentations
seed_nlabels, seed_labels, seed_stats, seed_centroids = cv2.connectedComponentsWithStats(seed_binary_map, None, None, None, 8, cv2.CV_32S)

# Get CC_STAT_AREA component as stats[label, COLUMN] 
seed_areas = seed_stats[1:,cv2.CC_STAT_AREA]

# Create a zero mask to reduce noise
seed_image_noise_reduce = np.zeros((seed_labels.shape), np.uint8)

seed_image_noise_reduce_copy = seed_image_noise_reduce.copy()
# Start to reduce noise
for i in range(0, seed_nlabels - 1):
    
    # If the segmented area is large, consider it is not a noise segmentation
    if seed_areas[i] >= 1:   
        seed_image_noise_reduce[seed_labels == i + 1] = 255

# Get contours of segmentations
seed_cnts = cv2.findContours(seed_image_noise_reduce, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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


# for loop for all contour
for seed_c in seed_cnts:
    
    # if it's too small, it might be noise, just ignore it
    if cv2.contourArea(seed_c) < 1:
        continue

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
        else:
            seed_endpoints_list.append([(seed_tltrX, seed_tltrY),(seed_blbrX, seed_blbrY)])


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

print("In totall frame quantity: ",frame-1)
# Generate csv files
#########################################################################################################################

# Create a list to store the lengths information
Microtubules_Length_Concatenated_to_Seeds = []

# Read the microtubules length concatanated to corresponding seeds in each frame, save them as sublist of label sequence list
for length in range(0,len(seed_correspond_microtubules_length),len(seed_endpoints_list)):
  length_group = seed_correspond_microtubules_length[length:length+len(seed_endpoints_list)]
  Microtubules_Length_Concatenated_to_Seeds.append(length_group)

# Store the information into csv file
file_csv_1 = open('Semantic_Segmentation/implementation/data_output/Microtubules_Lengths_with_Seed_Concatenation.csv','w',newline='')
writer_csv_1 = csv.writer(file_csv_1)
for concatenate_lengths_per_frame in Microtubules_Length_Concatenated_to_Seeds:
    writer_csv_1.writerow(concatenate_lengths_per_frame)


# Store the information into csv file
file_csv_2 = open('Semantic_Segmentation/implementation/data_output/Microtubules_Lengths_without_Concatenation.csv','w',newline='')
writer_csv_2 = csv.writer(file_csv_2)
for lengths_per_frame in Length_Micotubulues:
    writer_csv_2.writerow(lengths_per_frame)


# Generate visualization tools
#########################################################################################################################

# Vedio parameters setting
fps = 3
size = (seed_original.shape[0],seed_original.shape[1])
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter('Semantic_Segmentation/implementation/data_output/Microtubules_Lengths_Measurement.avi',fourcc,fps,size)

# Start to generate vedio
for visualize_measurement_frame in visualize_measurements_images_list:
    videoWriter.write(visualize_measurement_frame)

videoWriter.release()
cv2.destroyAllWindows()

# Set the seeds images and predictions saving path
prediction_composite_path = "Semantic_Segmentation/implementation/data_output/prediction_composite"

if os.path.exists(prediction_composite_path)==False:
    os.makedirs(prediction_composite_path)

# Save the predictions into the implementation_output folder
for n in range(0, frame-1):
    prediction_add = prediction_measurements_images_list[n]

    # Change the names of prediction for recognition and further easy to load
    implementation_prediction_save_path = "%s/prediction_composite_frame_%s.png"% (prediction_composite_path, (n+1))
    io.imsave(implementation_prediction_save_path, prediction_add)
