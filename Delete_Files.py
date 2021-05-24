import shutil
import os

shutil.rmtree("Data_Process/8bit_tiff_file")
os.mkdir("Data_Process/8bit_tiff_file")

shutil.rmtree("Data_Process/16bit_img_file")
os.mkdir("Data_Process/16bit_img_file")

shutil.rmtree("Data_Process/png_file")
os.mkdir("Data_Process/png_file")

shutil.rmtree("Semantic_Segmentation/implementation/input_image/")
os.mkdir("Semantic_Segmentation/implementation/input_image/")

shutil.rmtree("Semantic_Segmentation/implementation/prediction_image/")
os.mkdir("Semantic_Segmentation/implementation/prediction_image/")

shutil.rmtree("Semantic_Segmentation/implementation/prediction_seed/")
os.mkdir("Semantic_Segmentation/implementation/prediction_seed/")

shutil.rmtree("Semantic_Segmentation/implementation/seed_image/")
os.mkdir("Semantic_Segmentation/implementation/seed_image/")