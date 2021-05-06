import glob
import cv2
import pandas as pd
import os
import pydot
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from skimage import data_dir,io,transform,color

import segmentation_models as sm
import tensorflow as tf
from tensorflow import keras


#os.environ['DISPLAY'] = ':1'


# Creat test/score images list
score_images = []

# Creat prediction of test/score images list
test_outputs = []

# Set the test set images and predictions saving path
testset_file = "Semantic_Segmentation/training_data/image_test"
predict_file = "Semantic_Segmentation/training_data/test_predict_output/"

# Load the trained model
score_model = keras.models.load_model("Semantic_Segmentation/MT_1216_Semantic_Segmentation.h5", compile=False)

# Define a prediction process, with the input loading data are test set images and outputs are segmentation predictions
for directory_path in glob.glob(testset_file):
    
    # Check if the test set folder exist
    if os.path.exists(testset_file)==False:
        raise FileNotFoundError( 'No such file or directory:'+ testset_file)

    # Reading the images in the folder
    score_img_path = glob.glob(os.path.join(directory_path, '*.png'))

    # Make sure reading sequence of the images is correctly according to the name sequence of images
    score_img_path.sort()
    for l in score_img_path:
        
        # Read the images as RGB mode
        score_img = cv2.imread(l, cv2.IMREAD_COLOR)

        # Use reflect padding the images into size 1216x1216
        score_reflect_img = cv2.copyMakeBorder(score_img,8,8,8,8,cv2.BORDER_REFLECT)
        
        # Add up into score images list       
        score_images.append(score_reflect_img)

        # Convert list to array for machine learning processing
        score_reflect_img = np.array(score_reflect_img)

        # Expand the dimension of images for machine learning processing
        score_reflect_img = np.expand_dims(score_reflect_img, axis=0)

        # Use the trained model to predict segmentations
        score_prediction = score_model.predict(score_reflect_img)

        # Reshape the array into images
        score_prediction_image = score_prediction.reshape(1216, 1216)
        
        # Add up into prediction list
        test_outputs.append(score_prediction_image)


# Save the predictions into the test_output folder
for n in range(0, len(test_outputs)):
    test_prediction = test_outputs[n]

    # Change the names of prediction for recognition and further easy to load
    prediction_save_path = "%s/prediction_test_image_%s.png"% (predict_file, n)
    io.imsave(prediction_save_path, test_prediction)


