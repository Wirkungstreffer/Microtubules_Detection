Microtubules Detections

Readme

#########################################################################

Recommended hardware environment:

SWAP space >= 64GB

VRAM >= 8GB

Ram >= 32GB

hard drive space >= 80GB


Recommended software environment:

GPU driver: nvidia-driver-450

CUDA driver: V10.1

Anaconda python-3.7

Training semantic CNN could direct import conda env semantic_segmentation.yaml

Training DeepLabCut could direct import conda env DLC.yaml

#########################################################################

Guide:

1. Clone repository

2. Download data from the link https://drive.google.com/file/d/1oIn14fHrMmhm20Gkn8wGdebiwdsH62UZ/view?usp=sharing

3. Extract the data, move the images in folder "image" to the repository folder "Microtubules_Detection_Master/Semantic_Segmentation/training_data/images",
move the labels in folder "label" to the repository folder "Microtubules_Detection_Master/Semantic_Segmentation/training_data/labels".

4. Run the script Divide_Data_for_Training_and_Testing.py in folder "Microtubules_Detection_Master/Semantic_Segmentation/training_data/"

5. Run the script Augmentation.py in folder "Microtubules_Detection_Master/Semantic_Segmentation/training_data/".

6. Run the script Neural_Network_Training.py in folder "Microtubules_Detection_Master/Semantic_Segmentation/".

#########################################################################

To implement into new sequence after model is trained:

1. Put frames into folder "Microtubules_Detection_Master/Semantic_Segmentation/implementation/input_image", and seeds image into folder "Microtubules_Detection_Master/Semantic_Segmentation/implementation/seed_image".

2. Run Implementation_Prediction.py in folder "Microtubules_Detection_Master/Semantic_Segmentation/implementation/"

3. Run Length_Measurement.py in folder "Microtubules_Detection_Master/Semantic_Segmentation/implementation/"
#########################################################################

To transfer json file to image data, use console cd to the file, run $sh json_to_dataset.sh
