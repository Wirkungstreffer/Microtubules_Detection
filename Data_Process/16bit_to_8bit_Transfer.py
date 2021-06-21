import os
import cv2
import numpy as np

#### This function is for read 16-bit images and normalize further save them to 8-bit images ####
#### The input is images in folder "Data_Process/16bit_img_file" ####
#### Output will be store in folder "Data_Process/8bit_img_file" ####


# Increase contrast and brightness of image if needed
def contrast_img(image, c, b):  
    rows, cols = image.shape

    # Create a black image to be added up
    blank = np.zeros([rows, cols], image.dtype)

    # Add up brightness and contrast to the original image
    con_bri = cv2.addWeighted(src1 = image, alpha = c, src2 = blank, beta = 1-c, gamma = b)

    return con_bri



def read_directory(directory):
    # Define a image list, for further visualization
    array_of_img = []

    for filename in os.listdir(r"./"+ directory):
        # Reading 16-bit images
        tiff_image = cv2.imread(directory + "/" + filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        
        # Normalize them to 8-bit images
        img = cv2.normalize(tiff_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) 

        # Increase contrast and brightness
        img = contrast_img(img, 1.3, 20)

        # Check if the output folder exist
        if os.path.exists("Data_Process/8bit_tiff_file")==False:
            os.makedirs("Data_Process/8bit_tiff_file")
        
        # Save the transfered images into file "Data_Process/8bit_tiff_file"
        cv2.imwrite("Data_Process/8bit_tiff_file" + "/" + filename, img) 

        # Save images to list "array_of_img" in case for visualization 
        array_of_img.append(img)  
    
    return array_of_img

# Run the function
array_of_img = read_directory("Data_Process/16bit_img_file")  