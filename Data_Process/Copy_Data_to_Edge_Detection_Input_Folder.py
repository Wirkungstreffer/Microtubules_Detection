import shutil
import os

#### This function is to copy images into implementation input files ####
#### The input is images in folder "Data_Process/Video_Frames_8bit_png_files" and "Data_Process/Seeds_Image_8bit_png_files" ####
#### Output will be store in folder "Semantic_Segmentation/implementation/input_image/" and "Semantic_Segmentation/implementation/seed_image/" ####

original_frame_path = "Data_Process/Video_Frames_8bit_png_files"
original_seeds_path = "Data_Process/Seeds_Image_8bit_png_files"

copy_frame_path = "Edge_Detection/input_image/"
copy_seeds_path = "Edge_Detection/seed_image/"

# Check if the output folder exist
if os.path.exists(copy_frame_path)==False:
    os.makedirs(copy_frame_path)

if os.path.exists(copy_seeds_path)==False:
    os.makedirs(copy_seeds_path)


# Define a copy function
def duplicate_images(source_path, output_path):
    # Get the images list
    images = os.listdir(source_path)

    images.sort()
    
    # Start duplicate
    for image in images:
        
        # Designate output file directory
        output_image_path = os.path.join(source_path, image)

        if os.path.isfile(output_image_path):
            print (output_image_path)
            # Copy image
            shutil.copy(output_image_path, output_path)


# Run the copy function
duplicate_images(original_frame_path, copy_frame_path)
duplicate_images(original_seeds_path, copy_seeds_path)