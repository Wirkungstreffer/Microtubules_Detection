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


#Reload the prediction input images(better sorted version)
array_of_predict_input_image = [] 
def read_image(directory_name):
    imgList = os.listdir(r"./"+directory_name)
    imgList.sort(key=lambda x: int(x.replace("input_predict_","").split('.')[0]))
    
    for img_input in range(0, len(imgList)):
        filename = imgList[img_input]
        img = cv2.imread(directory_name + "/" + filename, 0)
        array_of_predict_input_image.append(img)
read_image("Semantic_Segmentation/implementation/prediction_image")
print("Input predictions quantity: " + str(len(array_of_predict_input_image)))



#Measure the lengths of Microtubules in testset

input_count = 1

for image in array_of_predict_input_image:

    #image = np.array(image, dtype=np.uint8)
    #Connect separate parts
    dilataion = cv2.dilate(image, None, iterations=1)
    erosion = cv2.erode(dilataion, None, iterations=1)

    implementation_input_original = implementation_input[input_count-1]

    #gray = cv2.cvtColor(erosion,cv2.COLOR_BGR2GRAY)
    # threshold
    #thresh = cv2.threshold(image,128,255,cv2.THRESH_BINARY)[1]

    # convert to binary by thresholding
    ret, binary_map = cv2.threshold(erosion,127,255,0)

    #binary_map = cv2.normalize(binary_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)

    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]

    image_noise_reduce = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if areas[i] >= 2:   #keep
            image_noise_reduce[labels == i + 1] = 255

    #image_noise_reduce_copy = image_noise_reduce.copy()
    # get contours
    cnts = cv2.findContours(image_noise_reduce, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    pixelsPerMetric = None

    # center point
    def midpoint(ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    boxes = []
    tltrX_list = []
    tltrY_list = []
    blbrX_list = []
    blbrY_list = []

    tlblX_list = []
    tlblY_list = []
    trbrX_list = []
    trbrY_list = []

    dA_list = []
    dB_list = []

    # for loop for all contour
    for c in cnts:
        # if it's too small, it might be noise, just ignore it
        if cv2.contourArea(c) < 20:
            continue

        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        
        box = perspective.order_points(box)

        boxes.append(box.astype("int"))

        #cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        #for (x, y) in box:
            #cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        tltrX_list.append(tltrX)
        tltrY_list.append(tltrY)
        blbrX_list.append(blbrX)
        blbrY_list.append(blbrY)

        tlblX_list.append(tlblX)
        tlblY_list.append(tlblY)
        trbrX_list.append(trbrX)
        trbrY_list.append(trbrY)


        #cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        #cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        #cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        #cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        #cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
        #cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)

        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        dA_list.append(dA)
        dB_list.append(dB)

        #if pixelsPerMetric is None:
        #	pixelsPerMetric = dB / imgsize[1]

        #dimA = dA / pixelsPerMetric
        #dimB = dB / pixelsPerMetric

        #cv2.putText(orig, "{:.1f}".format(dA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        #cv2.putText(orig, "{:.1f}".format(dB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)

        #cv2.imshow("Image", orig)
        #cv2.waitKey(0)

    tltrX_list = np.array(tltrX_list)
    tltrY_list = np.array(tltrY_list)
    blbrX_list = np.array(blbrX_list)
    blbrY_list = np.array(blbrY_list)

    tlblX_list = np.array(tlblX_list)
    tlblY_list = np.array(tlblY_list)
    trbrX_list = np.array(trbrX_list)
    trbrY_list = np.array(trbrY_list)

    dA_list = np.array(dA_list)
    #print(dA_list,dB_list)

    input_cnts_number = len(dA_list)
    print('The predicted image %s of testset detects Microtubules: %d' %(input_count, input_cnts_number))

    input_img = image_noise_reduce.copy()
    for i in range(len(tltrX_list)):
        cv2.line(input_img, (int(tltrX_list[i]), int(tltrY_list[i])), (int(blbrX_list[i]), int(blbrY_list[i])),(255, 0, 255), 2)
        cv2.line(input_img, (int(tlblX_list[i]), int(tlblY_list[i])), (int(trbrX_list[i]), int(trbrY_list[i])),(255, 0, 255), 2)
        cv2.putText(input_img, "{:.1f}".format(dA_list[i]), (int(tltrX_list[i] - 15), int(tltrY_list[i] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(input_img, "{:.1f}".format(dB_list[i]), (int(trbrX_list[i] + 10), int(trbrY_list[i])), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)

    #img_box = image.copy()
    #for k in range(len(boxes)):
    #	cv2.drawContours(img_box, [boxes[k].astype("int")], -1, (0, 255, 0), 2)

    #cv2.imshow("Image_box", img_box)

    #fig, ax = plt.subplots(ncols=2, figsize=(20, 8))
    #ax[0].imshow(implementation_input_original)
    #ax[0].set_title('input_image')
    #ax[0].axis('off')

    #ax[1].imshow(input_img)
    #ax[1].set_title('input_prediction')
    #ax[1].axis('off')

    input_count = input_count + 1

    cv2.imshow("Prediction + Measurement", input_img)
    cv2.waitKey(0)