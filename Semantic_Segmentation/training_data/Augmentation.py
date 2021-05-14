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


# Set the to be generated augmented image number
images_to_generate = 1500
#test_images_to_generate = 300

# Set the original images and labels folder path
images_path = "Semantic_Segmentation/training_data/images/" 
masks_path = "Semantic_Segmentation/training_data/labels/"

test_images_path = "Semantic_Segmentation/training_data/image_test/" 
test_masks_path = "Semantic_Segmentation/training_data/label_test/"

# Set the augmented images and labels saving path
img_augmented_path = "Semantic_Segmentation/training_data/images_aug/" 
msk_augmented_path = "Semantic_Segmentation/training_data/labels_aug/"

test_img_augmented_path = "Semantic_Segmentation/training_data/test_images_aug/" 
test_msk_augmented_path = "Semantic_Segmentation/training_data/test_labels_aug/"

# Load images and labels 
images = load_images(images_path)
masks = load_images(masks_path)

test_images = load_images(test_images_path)
test_masks = load_images(test_masks_path)

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

# Define a function which generate augmented data
def generate_augmented_data(aug_images_to_generate, images, masks, save_image_path, save_mask_path, image_tag):
    
    # variable to iterate till images_to_generate
    i = 1 

    # Use the loop to generate augmented images
    while i <= aug_images_to_generate: 
        
        # Pick a stochastic number to select the image & label in training set
        number = random.randint(0, len(images)-1)  
        image = images[number]
        mask = masks[number]

        # Print randomly selected image and label name
        print(image, mask)

        # Read the original image and label
        original_image = io.imread(image)
        original_mask = io.imread(mask)
        
        # Apply augmentation
        augmented = aug(image = original_image, mask = original_mask)
        transformed_image = augmented['image']
        transformed_mask = augmented['mask']

        # Change the names for recognition and further easy to load    
        new_image_path = "%s/augmented%s_image_%s.png" %(save_image_path, image_tag, i)
        new_mask_path = "%s/augmented%s_mask_%s.png" %(save_mask_path, image_tag, i)

        # Saving the augmented image & label
        io.imsave(new_image_path, transformed_image)
        io.imsave(new_mask_path, transformed_mask)
        
        i =i+1


# Generate augmented train set data
generate_augmented_data(images_to_generate, images, masks, img_augmented_path, msk_augmented_path,"_train")

# Generate augmented test set data
#generate_augmented_data(test_images_to_generate, test_images, test_masks, test_img_augmented_path, test_msk_augmented_path, "_test")