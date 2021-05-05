import glob
import cv2
from cv2 import drawContours
import os
import datetime
import pydot
os.environ['DISPLAY'] = ':1'
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import argparse
import imutils
from scipy import interpolate
import skimage
import skimage.morphology
import pwlf

import segmentation_models as sm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.utils import normalize
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from skimage import data_dir,io,transform,color

# Save predictions of seeds images
seeds_images = []
seeds_outputs = []

seed_image_file = "Semantic_Segmentation/implementation/seed_image/"
predict_seed_file = "Semantic_Segmentation/implementation/prediction_seed"

seed_model = keras.models.load_model("Semantic_Segmentation/MT_1216_Semantic_Segmentation_backup.h5", compile=False)

for directory_path in glob.glob(seed_image_file):
    seed_img_path = glob.glob(os.path.join(directory_path, '*.png'))
    seed_img_path.sort()
    for r in seed_img_path:
        seed_img = cv2.imread(r, cv2.IMREAD_COLOR)
        seed_reflect_img = cv2.copyMakeBorder(seed_img,8,8,8,8,cv2.BORDER_REFLECT)       
        seeds_images.append(seed_reflect_img)
        seed_reflect_img = np.array(seed_reflect_img)
        seed_reflect_img = np.expand_dims(seed_reflect_img, axis=0)
        seed_prediction = seed_model.predict(seed_reflect_img)
        seed_prediction_image = seed_prediction.reshape(1216, 1216)
        seeds_outputs.append(seed_prediction_image)


for q in range(0, len(seeds_outputs)):
    seeds_outputs_prediction = seeds_outputs[q]

    seeds_outputs_save_path = "%s/seed_predict_%s.png"% (predict_seed_file, q)
    io.imsave(seeds_outputs_save_path, seeds_outputs_prediction)




# Get predictions of implementation images as a list
implementation_input = []
implementation_predict_outputs = []

implementation_input_file = "Semantic_Segmentation/implementation/input_image/"
implementation_predict_file = "Semantic_Segmentation/implementation/prediction_image/"

input_imgName_List = []

implementation_model = keras.models.load_model("Semantic_Segmentation/MT_1216_Semantic_Segmentation_backup.h5", compile=False)
# Save predictions of implementation images as a list
for directory_path in glob.glob(implementation_input_file):
    implementation_img_path = glob.glob(os.path.join(directory_path, '*.png'))
    implementation_img_path.sort()
    for s in implementation_img_path:
        implementation_img = cv2.imread(s, cv2.IMREAD_COLOR)
        implementation_reflect_img = cv2.copyMakeBorder(implementation_img,8,8,8,8,cv2.BORDER_REFLECT)       
        implementation_input.append(implementation_reflect_img) #save input images as a list
        
        implementation_reflect_img = np.array(implementation_reflect_img)
        implementation_reflect_img = np.expand_dims(implementation_reflect_img, axis=0)
        implementation_prediction = implementation_model.predict(implementation_reflect_img)
        implementation_prediction_image = implementation_prediction.reshape(1216,1216)
        implementation_predict_outputs.append(implementation_prediction_image)

        input_im=Image.open(s)
        _,input_imgNamePNG=os.path.split(s)
        input_imgName,PNG=os.path.splitext(input_imgNamePNG)
        input_imgName_List.append(input_imgName)

print(input_imgName_List)

for t in range(0, len(implementation_predict_outputs)):
    implementation_input_prediction = implementation_predict_outputs[t]

    input_prediction_save_path = "%s/input_predict_%s.png"% (implementation_predict_file, t)
    io.imsave(input_prediction_save_path, implementation_input_prediction)
