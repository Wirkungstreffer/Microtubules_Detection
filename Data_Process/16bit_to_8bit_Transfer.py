import os
import cv2


# This function is for read 16-bit images and normalize further save them to 8-bit images 


def read_directory(directory):
    # Define a image list, for further visualization
    array_of_img = []

    for filename in os.listdir(r"./"+ directory):
        # Reading 16-bit images
        tiff_image = cv2.imread(directory + "/" + filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        
        # Normalize them to 8-bit images
        img = cv2.normalize(tiff_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) 

        # Save the transfered images into file "Data_Process/8bit_tiff_file"
        cv2.imwrite("Data_Process/8bit_tiff_file" + "/" + filename, img) 

        # Save images to list "array_of_img" in case for visualization 
        array_of_img.append(img)  
    return array_of_img

# Run the function
array_of_img = read_directory("Data_Process/16bit_img_file")  