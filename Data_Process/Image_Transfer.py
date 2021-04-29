import os
import cv2


# This function is for read and normalize images
array_of_img = []

def read_directory(directory):
    for filename in os.listdir(r"./"+ directory):
        tiff_image = cv2.imread(directory + "/" + filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        img = cv2.normalize(tiff_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        cv2.imwrite("Data_Process/8bit_tiff_file" + "/" + filename, img)
        
        array_of_img.append(img)

read_directory("Data_Process/16bit_img_file")