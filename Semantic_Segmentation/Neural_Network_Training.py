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
from keras.utils import normalize
from keras.utils import plot_model
from sklearn.model_selection import train_test_split


# If cv2.imshow() function dosen't work, use the followed line of code
os.environ['DISPLAY'] = ':1'

# Check if GPU is being used
tf.test.gpu_device_name()

#### The functions of this script is to train the convolutional neural network ####
#### The input data is the two files "/training_data/images_aug/" and "/training_data/labels_aug/" #### 
#### Outputs are plot of the hole neural network, trained model, validataion accuary and loss polts and logs for tenserboard if needed ####

# Functions Define
#########################################################################################################################

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
train_images = load_and_padding_images("Semantic_Segmentation/training_data/images_aug",3)

# Capture training augmented labels as a list
train_labels = load_and_padding_images("Semantic_Segmentation/training_data/labels_aug",1)

# Load test set images as a list
test_images = load_and_padding_images("Semantic_Segmentation/training_data/test_images_aug",3)

# Load test set labels as a list
test_labels = load_and_padding_images("Semantic_Segmentation/training_data/test_labels_aug",1)


# Load customary x_train and y_train variables
X = train_images
Y = train_labels

X_test = test_images
Y_test = test_labels

# Expand the dimension of label images for machine learning processing
Y = np.expand_dims(Y, axis=3)
Y_test = np.expand_dims(Y_test, axis=3) 

# Normalize the label images to make sure it is [0,1] binary images
Y = tf.keras.utils.normalize(Y)
Y_test = tf.keras.utils.normalize(Y_test)

# Sanity check, view the quantities of data sets and visualize few images
imgsize1 = X.shape
print(imgsize1)
imgsize2 = Y.shape
print(imgsize2)
imgsize3 = X_test.shape
print(imgsize3)
imgsize4 = Y_test.shape
print(imgsize4)

# Select random image in training dataset to visualize
import random
image_number = random.randint(0, len(X)-1)

cv2.imshow("X train",X[image_number])
cv2.waitKey(3000)

cv2.imshow("Y train",Y[image_number])
cv2.waitKey(3000)

test_image_number = random.randint(0, len(X_test)-1)

cv2.imshow("X test",X_test[image_number])
cv2.waitKey(3000)

cv2.imshow("Y test",Y_test[image_number])
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

# Training input: 1500 augmented images as training set and 60 images as test set with image size 1216x1216
# Tensor dimensions: 1216x1216x3 --- 608x608x64 --- 304x304x128 --- 152x152x256 --- 76x76x512 --- 38x38x1024 
history = model.fit(X, Y, 
                    batch_size = 2, 
                    #verbose = 1, 
                    epochs = 60, 
                    callbacks=callbacks,
                    validation_data=(X_test, Y_test)
                    #shuffle=False
                    )

# Save the model
model.save('./Semantic_Segmentation/MT_1216_Semantic_Segmentation.h5')


# Evaluate the model
#########################################################################################################################

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
plt.savefig("Semantic_Segmentation/loss.png")
plt.show()

# Plot the training and validation score at each epoch
acc = history.history['iou_score']
val_acc = history.history['val_iou_score']

plt.plot(epochs, acc, 'y', label='Training iou score')
plt.plot(epochs, val_acc, 'r', label='Validation iou score')
plt.title('Training and validation iou score')
plt.xlabel('Epochs')
plt.ylabel('IOU score')
plt.legend()
plt.savefig("Semantic_Segmentation/iou_score.png")
plt.show()


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
