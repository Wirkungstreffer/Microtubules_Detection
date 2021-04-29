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

tf.test.gpu_device_name()

#########################################################################################################################
# Define a simple Unet, for further design customed neural network structure.

def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
#Build the model
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv2D(75, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(75, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(100, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(100, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(200, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(200, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(400, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(400, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(1200, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(1200, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(400, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(400, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(400, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(200, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(200, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(200, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(100, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(100, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(100, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
     
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    opt = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[sm.metrics.iou_score])

    #model.compile(optimizer='adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])
    
    model.summary()
    
    return model

def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

#########################################################################################################################
# Date import and process

#Capture training image info as a list
train_images = []

for directory_path in glob.glob("Semantic_Segmentation/training_data/images_aug"):
    img_path = glob.glob(os.path.join(directory_path, '*.png'))
    img_path.sort()
    for i in img_path:
        img = cv2.imread(i, cv2.IMREAD_COLOR)
        reflect_img = cv2.copyMakeBorder(img,8,8,8,8,cv2.BORDER_REFLECT)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)       
        train_images.append(reflect_img)
#Convert list to array for machine learning processing        
train_images = np.array(train_images)

#Capture label info as a list
train_labels = [] 
for directory_path in glob.glob("Semantic_Segmentation/training_data/labels_aug"):
    label_path = glob.glob(os.path.join(directory_path, '*.png'))
    label_path.sort()
    for j in label_path:
        label = cv2.imread(j, 0)
        reflect_label = cv2.copyMakeBorder(label,8,8,8,8,cv2.BORDER_REFLECT)       
        #label = cv2.cvtColor(label, cv2.COLOR_RGB2BGR)
        train_labels.append(reflect_label)
#Convert list to array for machine learning processing          
train_labels = np.array(train_labels)


#Use customary x_train and y_train variables
X = train_images
#X = np.expand_dims(X, axis=3) 
Y = train_labels
Y = np.expand_dims(Y, axis=3) #May not be necessary..

#Splite data into training set and validation set
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

y_train = tf.keras.utils.normalize(y_train)
y_val = tf.keras.utils.normalize(y_val)

#Sanity check, view few mages
imgsize1 = x_train.shape
print(imgsize1)
imgsize2 = x_val.shape
print(imgsize2)
imgsize3 = y_train.shape
print(imgsize3)
imgsize4 = y_val.shape
print(imgsize4)

import random
image_number = random.randint(0, len(x_train)-1)

cv2.imshow("x train",x_train[image_number])
cv2.waitKey(3000)

#y_train_squeeze = y_train_nor[image_number]
#y_train_squeeze = y_train_squeeze[:,:,0]
cv2.imshow("y train",y_train[image_number])
cv2.waitKey(3000)


IMG_HEIGHT = train_images.shape[1]
IMG_WIDTH  = train_images.shape[2]
IMG_CHANNELS = train_images.shape[3]

# preprocess input
#x_train = preprocess_input(x_train)
#x_val = preprocess_input(x_val)

#model = get_model()

#########################################################################################################################
# Model von Segmentation_Models

# Encoder
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

# define optomizer
LR = 0.001
optim = keras.optimizers.Adam(LR)

# Model from the package semantic segmentation
model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model.compile(optimizer = optim, loss = sm.losses.bce_jaccard_loss, metrics = [sm.metrics.iou_score])

print("The model architecture:\n")
print(model.summary())

output_folder = "Semantic_Segmentation/model_output"
log_dir = os.path.join(output_folder,'logs_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

tf.keras.utils.plot_model(model,to_file=os.path.join(log_dir,'model.png'),show_shapes=True,show_layer_names=True)

#########################################################################################################################
# Training

#ModelCheckPoints
checkpointer = tf.keras.callbacks.ModelCheckpoint('MT_1216_Semantic_Segmentation.h5', verbose = 1, save_best_only = True)

callbacks = [#tf.keras.callbacks.EarlyStopping(patience = 4,monitor='val_iou_score'), 
             tf.keras.callbacks.TensorBoard(log_dir = './Semantic_Segmentation/logs')]
#Training

history = model.fit(x_train, y_train, 
                    batch_size = 2, 
                    #verbose = 1, 
                    epochs = 2, 
                    callbacks=callbacks,
                    validation_data=(x_val, y_val)
                    #shuffle=False
                    )

model.save('./Semantic_Segmentation/MT_1216_Semantic_Segmentation.h5')


# evaluate model
_, acc = model.evaluate(x_val, y_val, batch_size = 1)
print("Accuracy = ", (acc * 100.0), "%")

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('Semantic_Segmentation/loss.png')

#acc = history.history['acc']
acc = history.history['iou_score']
#val_acc = history.history['val_acc']
val_acc = history.history['val_iou_score']

plt.plot(epochs, acc, 'y', label='Training iou score')
plt.plot(epochs, val_acc, 'r', label='Validation iou score')
plt.title('Training and validation iou score')
plt.xlabel('Epochs')
plt.ylabel('IOU score')
plt.legend()
plt.show()
plt.savefig('Semantic_Segmentation/iou_score.png')


#########################################################################################################################
#Threshold adjustment for better result

#thre_model = keras.models.load_model("MT_1216_Semantic_Segmentation.h5", compile=False)
#y_pred = thre_model.predict(x_val)
#y_pred_thresholded = y_pred > 0.5
#intersection = np.logical_and(y_val, y_pred_thresholded)
#union = np.logical_or(y_val, y_pred_thresholded)
#iou_score = np.sum(intersection) / np.sum(union)
#print("IoU socre is: ", iou_score)

#########################################################################################################################
