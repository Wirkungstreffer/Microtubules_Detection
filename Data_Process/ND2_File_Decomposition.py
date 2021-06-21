from nd2reader import ND2Reader
import cv2
import os
import numpy as np

#### This function is for read ND2 files and normalize further save them to 8-bit png images ####
#### The input is images in folder "Data_Process/ND2_Video" and "Data_Process/ND2_Seeds" ####
#### Output will be store in folder "Data_Process/Video_Frames_8bit_png_files" and "Data_Process/Seeds_Image_8bit_png_files" ####


# Define the loading path
filepath = "Data_Process/ND2_Video"
filepath_seed = "Data_Process/ND2_Seeds"
output_path = "Data_Process/Video_Frames_8bit_png_files"
output_path_seed = "Data_Process/Seeds_Image_8bit_png_files"


# Loading the video frames and the name
for files in os.listdir(r"./"+ filepath):
  
  # Loading the ND2 file
  images = ND2Reader(filepath + "/" + files)
  
  # Get the ND2 files name
  filename = files.split(".nd2")[0]
  print("The video name is: " + filename)


# Loading the seeds image and the name
for files_seed in os.listdir(r"./"+ filepath_seed):
  
  # Loading the ND2 file
  seeds_image = ND2Reader(filepath_seed + "/" + files_seed)
  
  # Get the ND2 files name
  filename_seed = files_seed.split(".nd2")[0]
  print("The seeds image name is: " + filename_seed)


# Increase contrast and brightness of image if needed
def contrast_img(image, c, b):  
  rows, cols = image.shape

  # Create a black image to be added up
  blank = np.zeros([rows, cols], image.dtype)

  # Add up brightness and contrast to the original image
  con_bri = cv2.addWeighted(src1 = image, alpha = c, src2 = blank, beta = 1-c, gamma = b)

  return con_bri


# Use normalization to convert 16-bit images into 8-bit images, contrast and brightness are adjustable
# For contrast = 1.0 means original contrast, increas the contrast value to increase the contrast in image
# For brightness = 0 means original brightness, increas the brightness value to increase the brightness in image
def convert_16bit_to_8bit(images, filename, output_path, contrast, brightness):
  # Image number counter
  image_number = 1

  for image_16bit in images:
    # Normalize them to 8-bit images
    image_8bit = cv2.normalize(image_16bit, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) 

    # Increase contrast and brightness
    image_8bit = contrast_img(image_8bit, contrast, brightness)

    # Check if the output folder exist
    if os.path.exists(output_path)==False:
      os.makedirs(output_path)
    
    # Save the transfered images into file "Data_Process/Video_Frames_8bit_tiff"
    cv2.imwrite(output_path + "/" + filename + "_%03d.png"%(image_number), image_8bit)
    
    image_number = image_number + 1
    

# Run the convert function
convert_16bit_to_8bit(images, filename, output_path, 1.1, 0)
convert_16bit_to_8bit(seeds_image, filename_seed, output_path_seed, 1.1, 10)