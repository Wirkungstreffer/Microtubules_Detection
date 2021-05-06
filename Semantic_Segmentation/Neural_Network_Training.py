import glob
import cv2
import os
import datetime
import pydot
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image

import segmentation_models as sm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models
from tensorflow.keras.callbacks import TensorBoard
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.utils import normalize
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# If cv2.imshow() function dosen't work, use the followed line of code
os.environ['DISPLAY'] = ':1'

# Check if GPU is being used
tf.test.gpu_device_name()


# Functions Define

#########################################################################################################################

##### Define a simple Unet, for further design customed neural network structure #####
def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):

# Build the model, set the input as the image height, width and color channels
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Normalization of input data
    #s = Lambda(lambda x: x / 255)(inputs) 

    s = inputs

    # Encoder definition
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    # Decoder definition 
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
     
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    # Setting up optimizer, loss function and metrics
    #opt = tf.keras.optimizers.SGD(learning_rate=0.1)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['auccuracy'])

    #model.compile(optimizer='adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])
    
    model.summary()
    
    return model


# Load the model
def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

#model = get_model()



##### Date import and process #####

# Define a images loading and padding function, with the input loading data 1200x1200 png images and output 1216x1216 png images
def load_and_padding_images(image_file_directory, channel):

    # Create a image list
    image_set = []

    # Check if the input folder exist
    if os.path.exists(image_file_directory)==False:
        raise FileNotFoundError( 'No such file or directory:'+ image_file_directory)
        
    # Reading the images in the folder 
    for directory_path in glob.glob(image_file_directory):
        img_path = glob.glob(os.path.join(directory_path, '*.png'))

        # Make sure reading sequence of the images is correctly according to the name sequence of images
        img_path.sort()

        # Reading images
        for i in img_path:
            
            if channel == 3:
                # Read the images as RGB mode
                img = cv2.imread(i, cv2.IMREAD_COLOR)
            elif channel == 1:
                # Read the images as binary mode
                img = cv2.imread(i, 0)
            else:
                print("False channel input")

            # Use reflect padding the images into size 1216x1216
            reflect_img = cv2.copyMakeBorder(img,8,8,8,8,cv2.BORDER_REFLECT) 

            # Add up into images list     
            image_set.append(reflect_img)

    # Convert list to array for machine learning processing      
    image_set = np.array(image_set)
    
    return image_set



# Loading data

#########################################################################################################################


# Capture training augmented images as a list
train_val_images = load_and_padding_images("Semantic_Segmentation/training_data/images_aug",3)

# Capture training augmented labels as a list
train_val_labels = load_and_padding_images("Semantic_Segmentation/training_data/labels_aug",1)

# Load test set images as a list
test_images = load_and_padding_images("Semantic_Segmentation/training_data/image_test",3)

# Load test set labels as a list
test_labels = load_and_padding_images("Semantic_Segmentation/training_data/label_test",1)


# Load customary x_train and y_train variables
X = train_val_images
Y = train_val_labels

# Expand the dimension of label images for machine learning processing
Y = np.expand_dims(Y, axis=3)
test_labels = np.expand_dims(test_labels, axis=3) 

# Splite data into training set and validation set
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalize the label images to make sure it is [0,1] binary images
y_train = tf.keras.utils.normalize(y_train)
y_val = tf.keras.utils.normalize(y_val)

# Sanity check, view the quantities of data sets and visualize few images
imgsize1 = x_train.shape
print(imgsize1)
imgsize2 = x_val.shape
print(imgsize2)
imgsize3 = y_train.shape
print(imgsize3)
imgsize4 = y_val.shape
print(imgsize4)
imgsize5 = test_images.shape
print(imgsize5)
imgsize6 = test_labels.shape
print(imgsize6)

# Select random image in training and validation dataset to visualize
import random
image_number = random.randint(0, len(x_train)-1)

cv2.imshow("x train",x_train[image_number])
cv2.waitKey(3000)

cv2.imshow("y train",y_train[image_number])
cv2.waitKey(3000)



# Define and set up training model

#########################################################################################################################

# Model von Segmentation_Models
# Encoder
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

# define optomizer
LR = 0.001
optim = keras.optimizers.Adam(LR)

# Apply U-Net as network structure, set up the encoder weights, loss function and metrics
model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model.compile(optimizer = optim, loss = sm.losses.bce_jaccard_loss, metrics = [sm.metrics.iou_score])

# View the model summary
print("The model architecture:\n")
print(model.summary())

# Visualize the neural network structure and save the image to folder "Semantic_Segmentation/model_output"
output_folder = "Semantic_Segmentation/model_output"
log_dir = os.path.join(output_folder,'logs_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

tf.keras.utils.plot_model(model,to_file=os.path.join(log_dir,'model.png'),show_shapes=True,show_layer_names=True)



# Start Training

#########################################################################################################################

# Training Setting

# Set up ModelCheckPoints
checkpointer = tf.keras.callbacks.ModelCheckpoint('MT_1216_Semantic_Segmentation.h5', verbose = 1, save_best_only = True)

# Earlystop and Tensorboard can be added into callbacks if needed
callbacks = [#tf.keras.callbacks.EarlyStopping(patience = 4,monitor='val_iou_score'), 
             tf.keras.callbacks.TensorBoard(log_dir = './Semantic_Segmentation/logs')]

# Training input: 1200 images as training set and 300 images as validation set with image size 1216x1216 in RGB mode
# Tensor dimensions: 1216x1216x3 --- 608x608x64 --- 304x304x128 --- 152x152x256 --- 76x76x512 --- 38x38x1024 
history = model.fit(x_train, y_train, 
                    batch_size = 2, 
                    #verbose = 1, 
                    epochs = 2, 
                    callbacks=callbacks,
                    validation_data=(x_val, y_val)
                    #shuffle=False
                    )

# Save the model
model.save('./Semantic_Segmentation/MT_1216_Semantic_Segmentation.h5')


# Evaluate the model
_, acc = model.evaluate(test_images, test_labels, batch_size = 1)
print("Accuracy = ", (acc * 100.0), "%")

# Plot the training and validation losses at each epoch
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
plt.savefig("Semantic_Segmentation/loss.png")

# Plot the training and validation score at each epoch
acc = history.history['iou_score']
val_acc = history.history['val_iou_score']

plt.plot(epochs, acc, 'y', label='Training iou score')
plt.plot(epochs, val_acc, 'r', label='Validation iou score')
plt.title('Training and validation iou score')
plt.xlabel('Epochs')
plt.ylabel('IOU score')
plt.legend()
plt.show()
plt.savefig("Semantic_Segmentation/iou_score.png")


#########################################################################################################################
# Threshold adjustment for better result if needed

#thre_model = keras.models.load_model("MT_1216_Semantic_Segmentation.h5", compile=False)
#y_pred = thre_model.predict(test_images)
#y_pred_thresholded = y_pred > 0.5
#intersection = np.logical_and(test_labels, y_pred_thresholded)
#union = np.logical_or(test_labels, y_pred_thresholded)
#iou_score = np.sum(intersection) / np.sum(union)
#print("IoU socre is: ", iou_score)

#########################################################################################################################
