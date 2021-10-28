import numpy as np
import glob
import random
import os
import cv2

from matplotlib import pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage import io, img_as_ubyte
from scipy.ndimage import rotate
import albumentations as A

#### !!!! The script Divide_Data_for_Training_and_Testing.py must be run before runing this script !!!! ##### 

#### The functions of this script is to generate augmented images ####
#### Each image will generate several but same quantity randomly augmentations ####
#### The input data is the four files "/training_data/images/", "/training_data/labels/", "/training_data/image_test/" and "/training_data/label_test/"#### 
#### Output data will automatically store in "training_data/images_aug/", "training_data/labels_aug/", "training_data/test_images_aug/" and "training_data/test_labels_aug/"####

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



# Set the original images and labels folder path
images_path = "Semantic_Segmentation/training_data/images/" 
masks_path = "Semantic_Segmentation/training_data/labels/"


# Set the augmented images and labels saving path
img_augmented_path = "Semantic_Segmentation/training_data/images_aug/"
if os.path.exists(img_augmented_path)==False:
    os.makedirs(img_augmented_path)
msk_augmented_path = "Semantic_Segmentation/training_data/labels_aug/"
if os.path.exists(msk_augmented_path)==False:
    os.makedirs(msk_augmented_path)


# Load images and labels 
images = load_images(images_path)
masks = load_images(masks_path)

# Set up augmentation function
aug = A.Compose([
    A.VerticalFlip(p=0.5),              
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=1),
    A.Transpose(p=1),
    #A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    A.GridDistortion(p=1)
    ]
)


# Variants of generated augmentation count
count = 1

# Variants of image number of input data set
img_number = 0

# Pick a stochastic number to select the image & label in training set
while img_number < len(images):  
    
    image = images[img_number]
    mask = masks[img_number]

    # Print randomly selected image and label name
    print(image, mask)

    # Read the original image and label
    original_image = io.imread(image)
    original_mask = io.imread(mask)

    # For each image do the quantity of "loop" randomly augmentation
    loop = 0
    
    while loop <= 6:
        # Apply augmentation
        augmented = aug(image = original_image, mask = original_mask)
        transformed_image = augmented['image']
        transformed_mask = augmented['mask']

        # Change the names for recognition and further easy to load    
        new_image_path = "%s/augmented%s_image_%s.png" %(img_augmented_path, "_training", count)
        new_mask_path = "%s/augmented%s_mask_%s.png" %(msk_augmented_path, "_training", count)

        # Saving the augmented image & label
        io.imsave(new_image_path, transformed_image)
        io.imsave(new_mask_path, transformed_mask)
        
        count = count + 1
        loop = loop + 1
    
    img_number = img_number + 1
    


