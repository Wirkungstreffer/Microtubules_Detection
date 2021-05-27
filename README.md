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

2. Download data from the link https://drive.google.com/file/d/1lGBidoPvOS4NNZUAArCJNOjuyDw_x_bH/view?usp=sharing

3. Extract the data, move the images in folder "images" to the repository folder "Microtubules_Detection_Master/Semantic_Segmentation/training_data/images",
move the labels in folder "labels" to the repository folder "Microtubules_Detection_Master/Semantic_Segmentation/training_data/labels".

4. Run the script Divide_Data_for_Training_and_Testing.py in folder "Microtubules_Detection_Master/Semantic_Segmentation/training_data/"

5. Run the script Augmentation.py in folder "Microtubules_Detection_Master/Semantic_Segmentation/training_data/".

6. Run the script Neural_Network_Training.py in folder "Microtubules_Detection_Master/Semantic_Segmentation/".

#########################################################################

To implement into new sequence after model is trained:

(Make sure the trained network("MT_1216_Semantic_Segmentation.h5") is in the folder "Microtubules_Detection_Master/Semantic_Segmentation/")

1. Backup the last implementation data, run Delete_Last_Implementation_Data.py in folder "Microtubules_Detection_Master/Semantic_Segmentation/implementation/"

2. Put ND2 video into folder "Data_Process", and seeds image into folder "Microtubules_Detection_Master/Semantic_Segmentation/implementation/seed_image".

2. Run Implementation_Prediction.py in folder "Microtubules_Detection_Master/Semantic_Segmentation/implementation/"

3. Run Length_Measurement.py in folder "Microtubules_Detection_Master/Semantic_Segmentation/implementation/"

#########################################################################

To transfer json file to image data, use console cd to the file, run $sh json_to_dataset.sh
