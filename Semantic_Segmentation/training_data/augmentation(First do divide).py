import numpy as np
import glob
from matplotlib import pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage import io, img_as_ubyte
import random
import cv2
import os
from scipy.ndimage import rotate

import albumentations as A
images_to_generate = 10


images_path="Semantic_Segmentation/training_data/images/" #path to original images
masks_path = "Semantic_Segmentation/training_data/labels/"
img_augmented_path="Semantic_Segmentation/training_data/images_aug/" # path to store aumented images
msk_augmented_path="Semantic_Segmentation/training_data/labels_aug/" # 

images=[] # to store paths of images from folder
masks=[]

for directory_path in glob.glob(images_path):
    image_path = glob.glob(os.path.join(directory_path, '*.png'))
    image_path.sort()
    for i in image_path:  # read image name from folder and append its path into "images" array     
        images.append(i)

for directory_path in glob.glob(masks_path):
    label_path = glob.glob(os.path.join(directory_path, '*.png'))
    label_path.sort()
    for j in label_path:  # read image name from folder and append its path into "images" array     
        masks.append(j)


aug = A.Compose([
    A.VerticalFlip(p=0.5),              
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=1),
    A.Transpose(p=1),
    #A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    A.GridDistortion(p=1)
    ]
)

#random.seed(42)

i=1   # variable to iterate till images_to_generate


while i<=images_to_generate: 
    number = random.randint(0, len(images)-1)  #PIck a number to select an image & mask
    image = images[number]
    mask = masks[number]
    print(image, mask)
    #image=random.choice(images) #Randomly select an image name
    original_image = io.imread(image)
    original_mask = io.imread(mask)
    
    augmented = aug(image=original_image, mask=original_mask)
    transformed_image = augmented['image']
    transformed_mask = augmented['mask']

        
    new_image_path= "%s/augmented_image_%s.png" %(img_augmented_path, i)
    new_mask_path = "%s/augmented_mask_%s.png" %(msk_augmented_path, i)
    io.imsave(new_image_path, transformed_image)
    io.imsave(new_mask_path, transformed_mask)
    i =i+1