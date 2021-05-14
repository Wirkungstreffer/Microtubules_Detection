import glob
import cv2
from cv2 import drawContours
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import imutils
import skimage
import csv
import pwlf

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
seed_original = predict_seed_image[0]

# Convert prdiction image to binary map by thresholding
seed_ret, seed_binary_map = cv2.threshold(seed_erosion,127,255,0)

# Get the informations of segmentations
seed_nlabels, seed_labels, seed_stats, seed_centroids = cv2.connectedComponentsWithStats(seed_binary_map, None, None, None, 8, cv2.CV_32S)

# Get CC_STAT_AREA component as stats[label, COLUMN] 
seed_areas = seed_stats[1:,cv2.CC_STAT_AREA]

# Create a zero mask to reduce noise
seed_image_noise_reduce = np.zeros((seed_labels.shape), np.uint8)

# Start to reduce noise
for i in range(0, seed_nlabels - 1):
    
    # If the segmented area is large, consider it is not a noise segmentation
    if seed_areas[i] >= 2:   
        seed_image_noise_reduce[seed_labels == i + 1] = 255

# Get contours of segmentations
seed_cnts = cv2.findContours(seed_image_noise_reduce, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

seed_endpoints_list = []

# for loop for all contour
for seed_c in seed_cnts:
    
    # if it's too small, it might be noise, just ignore it
    if cv2.contourArea(seed_c) < 20:
        continue

    # Get the copy for later draw function
    seed_orig = seed_original.copy()

    # Use minimal area rectangular to box the segmentation
    seed_box = cv2.minAreaRect(seed_c)
    seed_box = cv2.cv.BoxPoints(seed_box) if imutils.is_cv2() else cv2.boxPoints(seed_box)
    seed_box = np.array(seed_box, dtype="int")
    seed_box = perspective.order_points(seed_box)

    seed_boxes.append(seed_box.astype("int"))

    # Get the midpoint of the length and width of the box
    (seed_tl, seed_tr, seed_br, seed_bl) = seed_box
    (seed_tltrX, seed_tltrY) = midpoint(seed_tl, seed_tr)
    (seed_blbrX, seed_blbrY) = midpoint(seed_bl, seed_br)

    (seed_tlblX, seed_tlblY) = midpoint(seed_tl, seed_bl)
    (seed_trbrX, seed_trbrY) = midpoint(seed_tr, seed_br)

    seed_endpoints_list.append([(seed_tlblX, seed_tlblY),(seed_trbrX, seed_trbrY)])

    # Add the points into the lists
    seed_tltrX_list.append(seed_tltrX)
    seed_tltrY_list.append(seed_tltrY)
    seed_blbrX_list.append(seed_blbrX)
    seed_blbrY_list.append(seed_blbrY)

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

# Convert the list to array for the further process
seed_tltrX_list = np.array(seed_tltrX_list)
seed_tltrY_list = np.array(seed_tltrY_list)
seed_blbrX_list = np.array(seed_blbrX_list)
seed_blbrY_list = np.array(seed_blbrY_list)

seed_tlblX_list = np.array(seed_tlblX_list)
seed_tlblY_list = np.array(seed_tlblY_list)
seed_trbrX_list = np.array(seed_trbrX_list)
seed_trbrY_list = np.array(seed_trbrY_list)

seed_dA_list = np.array(seed_dA_list)
seed_dB_list = np.array(seed_dB_list)
#print(dA_list,dB_list)

# Draw the length & width line and the number
seed_predict_img = seed_image_noise_reduce.copy()
for i in range(len(seed_tltrX_list)):
    cv2.line(seed_predict_img, (int(seed_tltrX_list[i]), int(seed_tltrY_list[i])), (int(seed_blbrX_list[i]), int(seed_blbrY_list[i])),(255, 0, 255), 2)
    cv2.line(seed_predict_img, (int(seed_tlblX_list[i]), int(seed_tlblY_list[i])), (int(seed_trbrX_list[i]), int(seed_trbrY_list[i])),(255, 0, 255), 2)
    cv2.putText(seed_predict_img, "{:.1f}".format(seed_dA_list[i]), (int(seed_tltrX_list[i] - 15), int(seed_tltrY_list[i] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(seed_predict_img, "{:.1f}".format(seed_dB_list[i]), (int(seed_trbrX_list[i] + 10), int(seed_trbrY_list[i])), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)


#cv2.imshow('seed',seed_predict_img)
#cv2.waitKey(0)

fig, ax = plt.subplots(ncols=2, figsize=(20, 8))
ax[0].imshow(seed_original)
ax[0].set_title('original seed image')
ax[0].axis('off')

ax[1].imshow(seed_predict_img)
ax[1].set_title('seed prediction and measurment')
ax[1].axis('off')

plt.savefig("Semantic_Segmentation/implementation/seed_measurement")
#plt.show()

#cv2.imshow("Prediction + Measurement", seed_predict_img)
#cv2.waitKey(0)


# Acquire microtubles position and calculate length information 
#########################################################################################################################

# Set a counter number of frame
frame = 1

# Create length and width recorder for each seed in each frame
seed_correspond_microtubules_length = []
seed_correspond_microtubules_width = []

# Set up deviation pixels tolerance
tolerance = 10

# Create list to store the number of length
Length_Micotubulues = []

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
        if areas[i] >= 2:   
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

    # for loop for all contour
    for c in cnts:
        
        # if it's too small, it might be noise, just ignore it
        if cv2.contourArea(c) < 20:
            continue

        # Get the copy for later draw function
        orig = image.copy()

        # Use minimal area rectangular to box the segmentation
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)

        boxes.append(box.astype("int"))

        # Get the midpoint of the length and width of the box
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

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

    
    # The lengths of microtubules that detected and measured in video frames
    Length_Micotubulues.append(dB_list)

    
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
        
        # Calculate the distance between each seeds and microtubules, save the microtubules have smaller distance than tolerance with seeds
        for mt in range(len(tlblX_list)):
            
            # Caculate the distance between seeds endpoints and microtubules endpoints
            distance_1 = dist.euclidean(seed_endpoint_1, (tlblX_list[mt], tlblY_list[mt]))
            distance_2 = dist.euclidean(seed_endpoint_1, (trbrX_list[mt], trbrY_list[mt]))
            distance_3 = dist.euclidean(seed_endpoint_2, (tlblX_list[mt], tlblY_list[mt]))
            distance_4 = dist.euclidean(seed_endpoint_2, (trbrX_list[mt], trbrY_list[mt]))

            # Save the microtubules index has smaller than tolerance
            if distance_1 < tolerance:
                correspond_mt.append(mt) 
            elif distance_2 < tolerance:
                correspond_mt.append(mt) 
            elif distance_3 < tolerance:
                correspond_mt.append(mt) 
            elif distance_4 < tolerance:
                correspond_mt.append(mt) 
            else:
                correspond_mt.append(len(tlblX_list) + 200)

        # Make the stored list into sublist according to each seed
        for mt_number in range(0,len(correspond_mt),len(tlblX_list)):
            number_group = correspond_mt[mt_number:mt_number+len(tlblX_list)]
            correspond_mt_per_seed.append(number_group)

        # Find the corresponding index for each seed
        for the_index in correspond_mt_per_seed:
            min_index_list.append(correspond_mt.index(min(the_index)))
            min_val_list.append(min(the_index))
        
        # Save the corresponding length information to the seed_correspond_microtubules lists
        for x in range(len(min_index_list)):
            if min_val_list[x] == len(tlblX_list) + 200 :
                seed_correspond_microtubules_width.append(0)
                seed_correspond_microtubules_length.append(0)
            else :
                seed_correspond_microtubules_width.append(dA_list[min_index_list[x]])
                seed_correspond_microtubules_length.append(dB_list[min_index_list[x]])



    # Draw the length & width line and the number
    predict_img = image_noise_reduce.copy()
    for i in range(len(tltrX_list)):
        cv2.line(predict_img, (int(tltrX_list[i]), int(tltrY_list[i])), (int(blbrX_list[i]), int(blbrY_list[i])),(255, 0, 255), 2)
        cv2.line(predict_img, (int(tlblX_list[i]), int(tlblY_list[i])), (int(trbrX_list[i]), int(trbrY_list[i])),(255, 0, 255), 2)
        cv2.putText(predict_img, "{:.1f}".format(dA_list[i]), (int(tltrX_list[i] - 15), int(tltrY_list[i] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(predict_img, "{:.1f}".format(dB_list[i]), (int(trbrX_list[i] + 10), int(trbrY_list[i])), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)


    #fig, ax = plt.subplots(ncols=2, figsize=(20, 8))
    #ax[0].imshow(input_original)
    #ax[0].set_title('original image')
    #ax[0].axis('off')

    #ax[1].imshow(predict_img)
    #ax[1].set_title('prediction and measurment')
    #ax[1].axis('off')

    #plt.show()

    frame = frame + 1

    #cv2.imshow("Prediction + Measurement", predict_img)
    #cv2.waitKey(0)


microtubules_length_all_frames = []
# Read the microtubules length for seeds in each frame, save them as sublist of label sequence list
for length in range(0,len(seed_correspond_microtubules_length),len(seed_endpoints_list)):
  length_group = seed_correspond_microtubules_length[length:length+len(seed_endpoints_list)]
  microtubules_length_all_frames.append(length_group)

print(microtubules_length_all_frames[0])
#file_csv = open('Semantic_Segmentation/implementation/Microtubules_Lengths.csv','w',newline='')
#writer_csv = csv.writer(file_csv)
#for lengths_per_frame in Length_Micotubulues:
#    writer_csv.writerow(lengths_per_frame)


