import numpy as np
import glob
import random
import cv2
import os

from matplotlib import pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage import io, img_as_ubyte
from scipy.ndimage import rotate
import albumentations as A

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
images_to_generate = 10

# Set the original images and labels folder path
images_path = "Semantic_Segmentation/training_data/images/" 
masks_path = "Semantic_Segmentation/training_data/labels/"

# Set the augmented images and labels saving path
img_augmented_path = "Semantic_Segmentation/training_data/images_aug/" 
msk_augmented_path = "Semantic_Segmentation/training_data/labels_aug/" 

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


# variable to iterate till images_to_generate
i = 1   

# Use the loop to generate augmented images
while i <= images_to_generate: 
    
    # Pick a stochastic number to select the image & label
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
    new_image_path = "%s/augmented_image_%s.png" %(img_augmented_path, i)
    new_mask_path = "%s/augmented_mask_%s.png" %(msk_augmented_path, i)

    # Saving the augmented image & label
    io.imsave(new_image_path, transformed_image)
    io.imsave(new_mask_path, transformed_mask)
    
    i =i+1