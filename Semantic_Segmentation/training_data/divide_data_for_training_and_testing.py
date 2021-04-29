import os
import random
import shutil
import glob
import cv2
os.environ['DISPLAY'] = ':1'


images_path="Semantic_Segmentation/training_data/images/" #path to all images
labels_path = "Semantic_Segmentation/training_data/labels/" #path to all labels

image_test_path="Semantic_Segmentation/training_data/image_test/" # path to store test images
label_test_path="Semantic_Segmentation/training_data/label_test/" # path to store test labels

images=[]
labels=[] 

for directory_path in glob.glob(images_path):
    image_path = glob.glob(os.path.join(directory_path, '*.png'))
    image_path.sort()
    for i in image_path:  # read image name from folder and append its path into "images" array     
        images.append(i)

for directory_path in glob.glob(labels_path):
    label_path = glob.glob(os.path.join(directory_path, '*.png'))
    label_path.sort()
    for j in label_path:  # read image name from folder and append its path into "labels" array     
        labels.append(j)


k=2   # number of test set images 
number = []

if k > len(images):
  print("The input number is larger than totall images quantity")
else:
  number = random.sample(range(0, len(images)-1), k)


for num in number: 
  image = images[num]
  label = labels[num]

  shutil.move(image, image_test_path)
  shutil.move(label, label_test_path)
