import glob
import cv2
import pandas as pd
from cv2 import drawContours
import os
import datetime
import pydot
os.environ['DISPLAY'] = ':1'
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


import segmentation_models as sm
import tensorflow as tf
from tensorflow import keras

from skimage import data_dir,io,transform,color



#Capture testset mask/label info as a list
score_masks = [] 
for directory_path in glob.glob("Semantic_Segmentation/training_data/label_test"):
    score_mask_path = glob.glob(os.path.join(directory_path, '*.png'))
    score_mask_path.sort()
    for m in score_mask_path:
        score_mask = cv2.imread(m, 0)
        score_reflect_mask = cv2.copyMakeBorder(score_mask,8,8,8,8,cv2.BORDER_REFLECT)       
        score_masks.append(score_reflect_mask)
#Convert list to array           
score_masks = np.array(score_masks)
print("Testset images quantity: " + str(len(score_masks)))

# Save predictions of testset images as a list
score_images = []
test_outputs = []

testset_file = "Semantic_Segmentation/training_data/image_test"
predict_file = "Semantic_Segmentation/training_data/test_predict_output/"

#for directory_path in glob.glob("Image_Data/semantic_segmentation_training/image_test"):
#    score_img_path = glob.glob(os.path.join(directory_path, '*.png'))
#    score_img_path.sort()
#    for n in score_img_path:
#        score_images.append(n)

imgName_List = []

score_model = keras.models.load_model("Semantic_Segmentation/MT_1216_Semantic_Segmentation.h5", compile=False)

for directory_path in glob.glob(testset_file):
    score_img_path = glob.glob(os.path.join(directory_path, '*.png'))
    score_img_path.sort()
    for l in score_img_path:
        score_img = cv2.imread(l, cv2.IMREAD_COLOR)
        score_reflect_img = cv2.copyMakeBorder(score_img,8,8,8,8,cv2.BORDER_REFLECT)       
        score_images.append(score_reflect_img)
        score_reflect_img = np.array(score_reflect_img)
        score_reflect_img = np.expand_dims(score_reflect_img, axis=0)
        score_prediction = score_model.predict(score_reflect_img)
        score_prediction_image = score_prediction.reshape(score_reflect_mask.shape)
        test_outputs.append(score_prediction_image)

        im=Image.open(l)
        _,imgNamePNG=os.path.split(l)
        imgName,PNG=os.path.splitext(imgNamePNG)
        imgName_List.append(imgName)

print(imgName_List)

for n in range(0, len(test_outputs)):
    test_prediction = test_outputs[n]

    prediction_save_path = "%s/prediction_test_image_%s.png"% (predict_file, n)
    io.imsave(prediction_save_path, test_prediction)