import os
import random
import shutil
import glob
import cv2
import numpy as np

#### The functions of this script is to split training and testing dataset ####
#### Total dataset has 26 sequence, each sequence has 12 images, this scrpit will randomly choose 5 sequence numbers of sequcences into testing dataset ####
#### The input data is the two files "/training_data/images/" and "/training_data/labels/" #### 
#### Output data will automatically store in "training_data/image_test/" and "training_data/label_test/" ####
#### The images names of training dataset and testing dataset will be stored into "training_data.txt" and "testing_data.txt" respectively ####


# Github cannot creat blank folder, so some txt files was create to maintain the folder, clear them would be better
def remove_inessential_files():
  if os.path.exists("Semantic_Segmentation/training_data/images/Training Images.txt"):
    os.remove("Semantic_Segmentation/training_data/images/Training Images.txt")
  if os.path.exists("Semantic_Segmentation/training_data/images_aug/Augmented Images.txt"):
    os.remove("Semantic_Segmentation/training_data/images_aug/Augmented Images.txt")
  if os.path.exists("Semantic_Segmentation/training_data/image_test/Test Image.txt"):
    os.remove("Semantic_Segmentation/training_data/image_test/Test Image.txt")
  if os.path.exists("Semantic_Segmentation/training_data/labels/Training Labels.txt"):
    os.remove("Semantic_Segmentation/training_data/labels/Training Labels.txt")
  if os.path.exists("Semantic_Segmentation/training_data/labels_aug/Augmented Labels.txt"):
    os.remove("Semantic_Segmentation/training_data/labels_aug/Augmented Labels.txt")
  if os.path.exists("Semantic_Segmentation/training_data/label_test/Label Testset.txt"):
    os.remove("Semantic_Segmentation/training_data/label_test/Label Testset.txt")
  if os.path.exists("Semantic_Segmentation/training_data/test_images_aug/test_images_aug.txt"):
    os.remove("Semantic_Segmentation/training_data/test_images_aug/test_images_aug.txt")
  if os.path.exists("Semantic_Segmentation/training_data/test_labels_aug/test_labels_aug.txt"):
    os.remove("Semantic_Segmentation/training_data/test_labels_aug/test_labels_aug.txt")

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


# Set the total images and labels folder path
images_path = "Semantic_Segmentation/training_data/images/" 
labels_path = "Semantic_Segmentation/training_data/labels/" 

# Set the test and validation set images and labels saving path
image_test_path = "Semantic_Segmentation/training_data/image_test/"
if os.path.exists(image_test_path)==False:
  os.makedirs(image_test_path) 
label_test_path = "Semantic_Segmentation/training_data/label_test/"
if os.path.exists(label_test_path)==False:
  os.makedirs(label_test_path) 

image_validation_path = "Semantic_Segmentation/training_data/image_validation/"
if os.path.exists(image_validation_path)==False:
  os.makedirs(image_validation_path) 
label_validation_path = "Semantic_Segmentation/training_data/label_validation/"
if os.path.exists(label_validation_path)==False:
  os.makedirs(label_validation_path) 

# Load images and labels
list_image_name, list_read_image = delete_end_str(images_path)
list_label_name, list_read_label = delete_end_str(labels_path)

# Each sequence has 12 images
image_quantity_each_sequence = 12

# Total sequence quantity in the hole dataset: 26
total_group_number = int(len(list_read_image)/image_quantity_each_sequence)

# Create image sequence list
images_sequence = []

# Read each 12 images, save them as sublist of image sequence list
for img in range(0,len(list_read_image),image_quantity_each_sequence):
  image_group = list_read_image[img:img+image_quantity_each_sequence]
  images_sequence.append(image_group)

# Create label sequence list
labels_sequence = []

# Read each 12 labels, save them as sublist of label sequence list
for lab in range(0,len(list_read_label),image_quantity_each_sequence):
  label_group = list_read_label[lab:lab+image_quantity_each_sequence]
  labels_sequence.append(label_group)


# Set the sequence number of test and validation set images
seq_validation = 5
seq_test = 5

# Creat a number list
seq_number_validation = []
seq_number_test = []

if max(seq_validation,seq_test) > total_group_number:

  # The test or validaiton set sequence number should smaller than total image sequence number
  print("The input sequence number is larger than totall sequence quantity")
else:

  # Non-repeatable randomly selet test and validation set images sequence quantity, store them in the seq_number list
  seq_number_validation = random.sample(range(0, total_group_number-1), seq_validation)
  seq_number_test = random.sample(range(0, total_group_number-1), seq_test)

# Define a file moving function which separate data set
def move_sequence_into_validationset(seq_number_val):
  for num_val in seq_number_val: 
    i=0
    # Select the images and labels correspending to the sequence number list
    sel_group_image_val = images_sequence[num_val]
    sel_group_label_val = labels_sequence[num_val]

    # Move the seleted images and labels sequence in to validation set folder
    while i < len(sel_group_image_val):
      shutil.move((images_sequence[num_val][i]+'.png'), image_validation_path)
      shutil.move((labels_sequence[num_val][i]+'.png'), label_validation_path)
      i = i + 1

# Define a file moving function which separate data set
def move_sequence_into_testset(seq_number_test):
  for num_test in seq_number_test: 
    j=0
    # Select the images and labels correspending to the sequence number list
    sel_group_image_test = images_sequence[num_test]
    sel_group_label_test = labels_sequence[num_test]

    # Move the seleted images and labels sequence in to test set folder
    while j < len(sel_group_image_test):
      shutil.move((images_sequence[num_test][j]+'.png'), image_test_path)
      shutil.move((labels_sequence[num_test][j]+'.png'), label_test_path)
      j = j + 1

# Implement dividing procedure
move_sequence_into_validationset(seq_number_validation)
move_sequence_into_testset(seq_number_test)

# Reading train dataset informations
training_image_name, training_image_path = delete_end_str(images_path)

# Save the training image names into "training_data.txt"
for train_file_name in training_image_path:
  train_file_name = train_file_name.strip("Semantic_Segmentation/training_data/images/")
  with open("Semantic_Segmentation/training_data/training_data.txt","a") as file:
    file.write(train_file_name + "\n")

# Reading test dataset informations
validation_image_name, validation_image_path = delete_end_str(image_validation_path)

# Save the testing names into "training_data.txt"
for test_file_name in validation_image_path:
  test_file_name = test_file_name.strip("Semantic_Segmentation/training_data/images/")
  with open("Semantic_Segmentation/training_data/validation_data.txt","a") as file:
    file.write(test_file_name + "\n")

# Reading test dataset informations
testing_image_name, testing_image_path = delete_end_str(image_test_path)

# Save the testing names into "training_data.txt"
for test_file_name in testing_image_path:
  test_file_name = test_file_name.strip("Semantic_Segmentation/training_data/images/")
  with open("Semantic_Segmentation/training_data/testing_data.txt","a") as file:
    file.write(test_file_name + "\n")