import cv2
import os
import imutils
from imutils import perspective
import numpy as np
import glob
from scipy.spatial import distance as dist
import csv

# If cv2.imshow() function dosen't work, use the followed line of code
os.environ['DISPLAY'] = ':1'

# Get center point
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

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

CNN_path = "Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200825_xb_reaction3_03um007_testset/CNN_Results/prediction_image/"
GT_path = "Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200825_xb_reaction3_03um007_testset/Ground_Truth/testset_label/"

error_csv_save_path = 'Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200825_xb_reaction3_03um007_testset/Error_List.csv'

array_of_CNN_image = load_images(CNN_path, 1)
array_of_GT_image = load_images(GT_path, 1)

#CNN_image = cv2.imread("Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200818_xb_reaction2_6um003_testset/CNN_Results/prediction_image/200818_xb_reaction2_6um003c1t010_prediction.png",0)
#GT_image = cv2.imread("Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200818_xb_reaction2_6um003_testset/Ground_Truth/testset_label/200818_xb_reaction2_6um003c1t010_label.png",0)

total_error_list = []
FN = 0
FP = 0
TP = 0
for number in range(len(array_of_CNN_image)):
    
    CNN_image = array_of_CNN_image[number]
    GT_image = array_of_GT_image[number]

    # Convert prdiction image to binary map by thresholding
    CNN_ret, CNN_binary_map = cv2.threshold(CNN_image,127,255,0)
    GT_ret, GT_binary_map = cv2.threshold(GT_image,1,255,0)

    CNN_cnts = cv2.findContours(CNN_binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    CNN_cnts = imutils.grab_contours(CNN_cnts)

    GT_cnts = cv2.findContours(GT_binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    GT_cnts = imutils.grab_contours(GT_cnts)

    error_list = []

    for GT_c in GT_cnts:

        orig = GT_image.copy()
        
        GT_box = cv2.minAreaRect(GT_c)
        GT_box = cv2.cv.BoxPoints(GT_box) if imutils.is_cv2() else cv2.boxPoints(GT_box)
        GT_box = np.array(GT_box, dtype="int")
        GT_box = perspective.order_points(GT_box)

        # Get the midpoint of the length and width of the box
        (GT_tl, GT_tr, GT_br, GT_bl) = GT_box

        # Midpoints of the width sides of box
        (GT_tltrX, GT_tltrY) = midpoint(GT_tl, GT_tr)
        (GT_blbrX, GT_blbrY) = midpoint(GT_bl, GT_br)

        # Midpoints of the length sides of box
        (GT_tlblX, GT_tlblY) = midpoint(GT_tl, GT_bl)
        (GT_trbrX, GT_trbrY) = midpoint(GT_tr, GT_br)

        GT_dB = dist.euclidean((GT_tlblX, GT_tlblY), (GT_trbrX, GT_trbrY))
        
        (GT_center_point_X, GT_center_point_Y) = midpoint((GT_tlblX, GT_tlblY) ,(GT_trbrX, GT_trbrY))

        for CNN_c in CNN_cnts:
            CNN_box = cv2.minAreaRect(CNN_c)
            CNN_box = cv2.cv.BoxPoints(CNN_box) if imutils.is_cv2() else cv2.boxPoints(CNN_box)
            CNN_box = np.array(CNN_box, dtype="int")
            CNN_box = perspective.order_points(CNN_box)

            # Get the midpoint of the length and width of the box
            (CNN_tl, CNN_tr, CNN_br, CNN_bl) = CNN_box
            # Midpoints of the width sides of box
            (CNN_tltrX, CNN_tltrY) = midpoint(CNN_tl, CNN_tr)
            (CNN_blbrX, CNN_blbrY) = midpoint(CNN_bl, CNN_br)

            # Midpoints of the length sides of box
            (CNN_tlblX, CNN_tlblY) = midpoint(CNN_tl, CNN_bl)
            (CNN_trbrX, CNN_trbrY) = midpoint(CNN_tr, CNN_br)

            CNN_dB = dist.euclidean((CNN_tlblX,  CNN_tlblY), (CNN_trbrX, CNN_trbrY))

            (CNN_center_point_X, CNN_center_point_Y) = midpoint((CNN_tlblX, CNN_tlblY),(CNN_trbrX, CNN_trbrY))

            segmentation_distance = dist.euclidean((GT_center_point_X, GT_center_point_Y),(CNN_center_point_X, CNN_center_point_Y))

            
            if segmentation_distance <= 10:
                error = abs(GT_dB - CNN_dB)
                error_list.append(error)

    TP = TP + len(error_list)

    if len(GT_cnts) > len(error_list):
        FN = FN + (len(GT_cnts) - len(error_list))

    if len(CNN_cnts) > len(error_list):
        FP = FP + (len(CNN_cnts) - len(error_list))

    total_error_list.append(error_list)

print("True Positive:",TP)
print("False Positive",FP)
print("False Negative:",FN)

# Store the information into csv file
file_csv_1 = open(error_csv_save_path,'w',newline='')
writer_csv_1 = csv.writer(file_csv_1)
for errors in total_error_list:
    writer_csv_1.writerow(errors)