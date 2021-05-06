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

2. Download data from the link https://drive.google.com/file/d/1uo9Oq5h7p7Uw4NPF8suhekrteepqilI4/view?usp=sharing

3. Extract the data, move the images in folder "Image" to the repository folder "Microtubules_Detection_Master/Semantic_Segmentation/training_data/images",
move the labels in folder "Label" to the repository folder "Microtubules_Detection_Master/Semantic_Segmentation/training_data/labels".

4. Run the script Divide_Data_for_Training_and_Testing.py in folder "Microtubules_Detection_Master/Semantic_Segmentation/training_data/"

5. Run the Augmentation.py in folder "Microtubules_Detection_Master/Semantic_Segmentation/training_data/".

6. Run the Neural_Network_Training.py in folder "Microtubules_Detection_Master/Semantic_Segmentation/".

#########################################################################

To transfer json file to image data, use console cd to the file, run $sh json_to_dataset.sh
