import glob
import cv2
import os
import numpy as np
from PIL import Image
from numpy.lib.type_check import imag
from skimage import io

import tensorflow as tf
from tensorflow import keras


#### This script will get the predictions of input images ####
#### The inputs frame images of microtubules should be put in the folder "input_image" ####
#### The seed image should be put in the folder "seed_image" ####
#### Outputs of frame images predictions will be stored in folder "prediction_image" ####
#### Outputs of seed image predictions will be stored in folder "prediction_seed" #### 

# Github cannot creat blank folder, so some txt files was create to maintain the folder, clear them would be better
def remove_inessential_files():
  if os.path.exists("Semantic_Segmentation/implementation/input_image/input_image.txt"):
    os.remove("Semantic_Segmentation/implementation/input_image/input_image.txt")
  if os.path.exists("Semantic_Segmentation/implementation/prediction_image/prediction_image.txt"):
    os.remove("Semantic_Segmentation/implementation/prediction_image/prediction_image.txt")
  if os.path.exists("Semantic_Segmentation/implementation/prediction_seed/prediction_seed.txt"):
    os.remove("Semantic_Segmentation/implementation/prediction_seed/prediction_seed.txt")
  if os.path.exists("Semantic_Segmentation/implementation/seed_image/seed_image.txt"):
    os.remove("Semantic_Segmentation/implementation/seed_image/seed_image.txt")

# Clear the redundant txt files
remove_inessential_files()

# Define a images path and name reading function
def delete_end_str(path):

  # Check if the input folder exist
  if os.path.exists(path)==False:
    raise FileNotFoundError( 'No such file or directory:'+ path)
  
  # Create image path list and image name list
  list_img_name = []
  list_read_img = []
  
  # Start reading image path and name
  filelist = os.listdir(path)
  for filename in filelist:
    filename = path + filename
    
    # Add image name to list
    list_img_name.append(filename)
    
    # Make sure the names is in order
    list_img_name.sort()

    # Get the image path
    new_file_name = filename.split(".png")[0]
    
    # Add image path to list
    list_read_img.append(new_file_name)
    
    # Make sure the path is in order
    list_read_img.sort()
  
  return list_img_name, list_read_img


# Define a prediction process, with the input loading data are implementation images and outputs are segmentation predictions
def load_and_predict(implementation_input_file, trained_model):

  # Load the trained model
  implementation_model = trained_model

  # Creat implementation images list
  implementation_inputs = []

  # Creat prediction of implementation images list
  implementation_outputs = []

  for directory_path in glob.glob(implementation_input_file):

    # Check if the implementation folder exist
    if os.path.exists(implementation_input_file)==False:
      raise FileNotFoundError( 'No such file or directory:'+ implementation_input_file)

    # Reading the images in the folder
    implementation_img_path = glob.glob(os.path.join(directory_path, '*.png'))

    # Make sure reading sequence of the images is correctly according to the name sequence of images
    implementation_img_path.sort()
    for s in implementation_img_path:
        
      # Read the images as RGB mode
      implementation_img = cv2.imread(s, cv2.IMREAD_COLOR)
      
      # Get the image size
      image_size_x = implementation_img.shape[0]
      image_size_y = implementation_img.shape[1]

      # Calculate the compensate pixels to make image 32 divisible
      if image_size_x % 32 == 0:
        compensate_x = 0
      else:
        compensate_x = int(image_size_x/32+1)*32 - image_size_x

      if image_size_y % 32 == 0:
        compensate_y = 0
      else:
        compensate_y = int(image_size_y/32+1)*32 - image_size_y

      # Use reflect padding the images into 32 divisible size
      implementation_reflect_img = cv2.copyMakeBorder(implementation_img,0,compensate_x,0,compensate_y,cv2.BORDER_REFLECT)

      # Store the compensated image size
      expanded_size_x =  implementation_reflect_img.shape[0]
      expanded_size_y =  implementation_reflect_img.shape[1]     
      
      # Add up into implementation images list 
      implementation_inputs.append(implementation_reflect_img)
      
      # Convert list to array for machine learning processing
      implementation_reflect_img = np.array(implementation_reflect_img)

      # Expand the dimension of images for machine learning processing
      implementation_reflect_img = np.expand_dims(implementation_reflect_img, axis=0)

      # Use the trained model to predict segmentations
      implementation_prediction = implementation_model.predict(implementation_reflect_img)

      # Reshape the array into image
      implementation_prediction_image = implementation_prediction.reshape(expanded_size_x, expanded_size_y)

      # Crop the output image into the original size
      implementation_prediction_image_cropped = implementation_prediction_image[0:image_size_x, 0:image_size_y]

      # Add up into prediction list
      implementation_outputs.append(implementation_prediction_image_cropped)
  
  return implementation_inputs, implementation_outputs



# Set the seeds images and predictions saving path
seed_image_file = "Semantic_Segmentation/implementation/seed_image/"

predict_seed_file = "Semantic_Segmentation/implementation/prediction_seed"
if os.path.exists(predict_seed_file)==False:
    os.makedirs(predict_seed_file)

# Load the trained model
seed_model = keras.models.load_model("Semantic_Segmentation/MT_1216_Semantic_Segmentation.h5", compile=False)

# Load seed image and use trained model to predict segmentations
seed_images, seed_outputs = load_and_predict(seed_image_file, seed_model)

# Get the seeds images names
seed_img_name, seed_read_path = delete_end_str(seed_image_file)

# Create a name list
seed_name_list = []

# Save the seeds image names into name list
for seed_file_name in seed_read_path:
  seed_file_name = seed_file_name.strip("Semantic_Segmentation/implementation/seed_image/")
  seed_name_list.append(seed_file_name)

# Save the predictions into the seeds_output folder
for n in range(0, len(seed_outputs)):
  seed_prediction = seed_outputs[n]

  # Change the names of prediction for recognition and further easy to load
  seeds_prediction_save_path = "%s/%s_prediction.png"% (predict_seed_file, seed_name_list[n])
  io.imsave(seeds_prediction_save_path, seed_prediction)



# Set the implementation images and predictions saving path
implementation_input_file = "Semantic_Segmentation/implementation/input_image/"

implementation_predict_file = "Semantic_Segmentation/implementation/prediction_image/"
if os.path.exists(implementation_predict_file)==False:
    os.makedirs(implementation_predict_file)

# Load the trained model
implementation_model = keras.models.load_model("Semantic_Segmentation/MT_1216_Semantic_Segmentation.h5", compile=False)

# Load implementation image and use trained model to predict segmentations
implementation_inputs, implementation_outputs = load_and_predict(implementation_input_file, implementation_model)

# Get the implementation images names
implementation_img_name, implementation_read_path = delete_end_str(implementation_input_file)

# Create a name list
implementation_name_list = []

# Save the testing names into name list
for implementation_file_name in implementation_read_path:
  implementation_file_name = implementation_file_name.strip("Semantic_Segmentation/implementation/input_image/")
  implementation_name_list.append(implementation_file_name)

# Save the predictions into the implementation_output folder
for n in range(0, len(implementation_outputs)):
  implementation_prediction = implementation_outputs[n]

  # Change the names of prediction for recognition and further easy to load
  implementation_prediction_save_path = "%s/%s_prediction.png"% (implementation_predict_file, implementation_name_list[n])
  io.imsave(implementation_prediction_save_path, implementation_prediction)