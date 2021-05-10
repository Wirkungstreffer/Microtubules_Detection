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
original_images_path = "Semantic_Segmentation/training_data/image_test/"
predict_image_path = "Semantic_Segmentation/training_data/test_predict_output/"

# Load original images and predicted images
array_of_input_image = load_images(original_images_path, 3)
array_of_predict_input_image = load_images(predict_image_path, 1)

# Set a counter number
input_count = 1

# Measure the lengths of microtubules segmentation
for image in array_of_predict_input_image:

    # Connect separate parts
    dilataion = cv2.dilate(image, None, iterations=1)
    erosion = cv2.erode(dilataion, None, iterations=1)

    # Read the original image
    input_original = array_of_input_image[input_count-1]

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
        
        # If the segmented area is large than 2, consider it is not a noise segmentation
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

        orig = image.copy()

        # Use minimal area rectangular to box the segmentation
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)

        boxes.append(box.astype("int"))

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
    #print(dA_list,dB_list)

    # Draw the length & width line and the number
    input_img = image_noise_reduce.copy()
    for i in range(len(tltrX_list)):
        cv2.line(input_img, (int(tltrX_list[i]), int(tltrY_list[i])), (int(blbrX_list[i]), int(blbrY_list[i])),(255, 0, 255), 2)
        cv2.line(input_img, (int(tlblX_list[i]), int(tlblY_list[i])), (int(trbrX_list[i]), int(trbrY_list[i])),(255, 0, 255), 2)
        cv2.putText(input_img, "{:.1f}".format(dA_list[i]), (int(tltrX_list[i] - 15), int(tltrY_list[i] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(input_img, "{:.1f}".format(dB_list[i]), (int(trbrX_list[i] + 10), int(trbrY_list[i])), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)


    fig, ax = plt.subplots(ncols=2, figsize=(20, 8))
    ax[0].imshow(input_original)
    ax[0].set_title('original image')
    ax[0].axis('off')

    ax[1].imshow(input_img)
    ax[1].set_title('prediction and measurment')
    ax[1].axis('off')

    #plt.show()

    input_count = input_count + 1

    cv2.imshow("Prediction + Measurement", input_img)
    cv2.waitKey(0)