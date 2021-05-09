import os 
import numpy as np
import pandas as pd
import glob,os
from PIL import Image

#### This function is for read 8-bit tiff images and convert to png images ####
#### The input is images in folder "Data_Process/8bit_img_file" ####
#### Output will be store in folder "Data_Process/png_file" ####

# Define the input tiff images folder and output png images folder
TIFF_IMAGE_FILE=r'Data_Process/8bit_tiff_file'
PNG_IMAGE_FILE=r'Data_Process/png_file'


# This function is for converting image format  
def change_image_format_batch(src_path, tar_path, fmt_in, fmt_out ):
  
  # Check if the input folder exist
  if os.path.exists(src_path)==False:
    raise FileNotFoundError( 'No such file or directory:'+ src_path)
  
  # Create a dictionary of input folder
  img_dict = dict()
  directorys = [ subpath for subpath in os.listdir(src_path) if   os.path.isdir( os.path.join(src_path,subpath) )   ]

  # Reading the names and format of input images
  if len(directorys)==0:
    imgPaths=glob.glob(os.path.join(src_path,'*.'+ fmt_in))
    
    # If output folder is not exist, create such folder
    if os.path.exists(tar_path)==False:
      os.makedirs(tar_path)
    
    # Separate the names and the format of images, store the names
    for imgPath in imgPaths:
      im=Image.open(imgPath)
      _,imgNamePNG=os.path.split(imgPath)
      imgName,PNG=os.path.splitext(imgNamePNG)
      im.save(os.path.join(tar_path,imgName+'.'+ fmt_out))
    return

  # Check the subfolders of input folder in the dictionary
  for subdir in directorys:
    img_dict[subdir]=glob.glob(os.path.join(src_path,subdir,'*.'+ fmt_in))
  
  # If the subfolders is not exist in output folder, create such subfolders
  if os.path.exists(tar_path)==False:
    os.makedirs(tar_path)
  
  # Save the names and changed format into output folder
  for subdir,imgPaths in img_dict.items():
    newLongdir=os.path.join(tar_path,subdir)
    
    # Check if the output folder exist
    if os.path.exists(newLongdir)==False:
      os.makedirs(newLongdir)
    
    # Save the names of images with the wanted format
    for imgPath in imgPaths:
      im=Image.open(imgPath)
      _,imgNamePNG=os.path.split(imgPath)
      imgName,PNG=os.path.splitext(imgNamePNG)
      im.save(os.path.join(tar_path,subdir,imgName+'.'+ fmt_out))



# convert tiff into png 
change_image_format_batch(TIFF_IMAGE_FILE,PNG_IMAGE_FILE,'tif','png')