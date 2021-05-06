import os
import random
import shutil
import glob
import cv2




# Define a images loading function
def load_images(image_file_directory):
    # Create a image list
    images = []

    # Check if the input folder exist
    if os.path.exists(image_file_directory)==False:
        raise FileNotFoundError( 'No such file or directory:'+ image_file_directory)

    # Reading the images in the folder 
    for directory_path in glob.glob(image_file_directory):
        image_path = glob.glob(os.path.join(directory_path, '*.png'))

        # Make sure reading sequence of the images is correctly according to the name sequence of images
        image_path.sort()

        # Reading images, dd up into images list
        for i in image_path:     
            images.append(i)

    return images


# Set the total images and labels folder path
images_path = "Semantic_Segmentation/training_data/images/" 
labels_path = "Semantic_Segmentation/training_data/labels/" 

# Set the test set images and labels saving path
image_test_path = "Semantic_Segmentation/training_data/image_test/" 
label_test_path = "Semantic_Segmentation/training_data/label_test/" 

# Load images and labels
images = load_images(images_path)
labels = load_images(labels_path)

# Set the number of test set images
k=37 

# Creat a number list
number = []

if k > len(images):

  # The test set number should smaller than total image number
  print("The input number is larger than totall images quantity")
else:

  # Non-repeatable randomly selet test set images quantity k, store them in the number list
  number = random.sample(range(0, len(images)-1), k)


for num in number: 

  # Select the images and labels correspending to the number list
  image = images[num]
  label = labels[num]

  # Move the seleted images and labels in to test set folder
  shutil.move(image, image_test_path)
  shutil.move(label, label_test_path)
