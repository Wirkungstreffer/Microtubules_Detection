import shutil
import os

#### !!!! Before run this script, please bakcup all the necessary image data, make sure the data is stored in other folders !!!! ####
 
#### This function is to delete all the data from last implementation ####

shutil.rmtree("Data_Process/ND2_Seeds")
os.mkdir("Data_Process/ND2_Seeds")

shutil.rmtree("Data_Process/ND2_Video")
os.mkdir("Data_Process/ND2_Video")

shutil.rmtree("Data_Process/Seeds_Image_8bit_png_files")
os.mkdir("Data_Process/Seeds_Image_8bit_png_files")

shutil.rmtree("Data_Process/Video_Frames_8bit_png_files")
os.mkdir("Data_Process/Video_Frames_8bit_png_files")

shutil.rmtree("Semantic_Segmentation/implementation/input_image/")
os.mkdir("Semantic_Segmentation/implementation/input_image/")

shutil.rmtree("Semantic_Segmentation/implementation/prediction_image/")
os.mkdir("Semantic_Segmentation/implementation/prediction_image/")

shutil.rmtree("Semantic_Segmentation/implementation/prediction_seed/")
os.mkdir("Semantic_Segmentation/implementation/prediction_seed/")

shutil.rmtree("Semantic_Segmentation/implementation/seed_image/")
os.mkdir("Semantic_Segmentation/implementation/seed_image/")