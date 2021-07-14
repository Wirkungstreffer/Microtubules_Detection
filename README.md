Microtubules Detections

Readme

#############################################################################################

Recommended hardware environment:

SWAP space >= 64GB

VRAM >= 8GB

Ram >= 32GB

hard drive space >= 80GB


Recommended software environment:

GPU driver: nvidia-driver-450 or nvidia-driver-460

CUDA driver: V10.1

Anaconda python-3.7

Training semantic CNN could direct import conda env semantic_segmentation.yaml

Training DeepLabCut could direct import conda env DLC.yaml

#############################################################################################

Neural network training guide:

1. Clone repository

2. Download data from the link: https://drive.google.com/file/d/1lGBidoPvOS4NNZUAArCJNOjuyDw_x_bH/view?usp=sharing

3. Extract the data, move the images in folder "images" to the repository folder "Microtubules_Detection-master/Semantic_Segmentation/training_data/images",
move the labels in folder "labels" to the repository folder "Microtubules_Detection-master/Semantic_Segmentation/training_data/labels".

4. Run the script Divide_Data_for_Training_and_Testing.py in folder "Microtubules_Detection-master/Semantic_Segmentation/training_data/"

5. Run the script Augmentation.py in folder "Microtubules_Detection-master/Semantic_Segmentation/training_data/".

6. Run the script Neural_Network_Training.py in folder "Microtubules_Detection-master/Semantic_Segmentation/".

#############################################################################################

To implement into new sequence after model is trained:

(If needed)Trained network model download link: https://drive.google.com/file/d/1DcdysUrOZF6n4mP0bd7U7637yIShauK7/view?usp=sharing
(Make sure the trained network("MT_1216_Semantic_Segmentation.h5") is in the folder "Microtubules_Detection-master/Semantic_Segmentation/")

1. Backup the last implementation data, run Delete_Last_Implementation_Data.py in folder "Microtubules_Detection-master/Semantic_Segmentation/implementation/"

2. Put ND2 video into folder "Microtubules_Detection-master/Data_Process/ND2_Video/", and seeds ND2 file into folder "Microtubules_Detection-master/Data_Process/ND2_Seeds/".

3. Run ND2_File_Decomposition.py in folder "Microtubules_Detection-master/Data_Process/". (Contrast and brightness are adjustable depand on the input data)

4. Run Duplicate_Data_to_Implementation_Input_Folder.py in folder "Microtubules_Detection-master/Data_Process/"

5. Run Implementation_Prediction.py in folder "Microtubules_Detection-master/Semantic_Segmentation/implementation/"

6. Run Length_Measurement.py in folder "Microtubules_Detection-master/Semantic_Segmentation/implementation/"

7. Run Automatic_Linear_Regression.py in folder "Microtubules_Detection-master/Semantic_Segmentation/implementation/"

8. (Optional) Run Manually_Linear_Regression_Correction.py in folder "Microtubules_Detection-master/Semantic_Segmentation/implementation/" if the automatic detected linear regressions are inaccurate.

All the output data is in the folder "Microtubules_Detection-master/Semantic_Segmentation/implementation/data_output"

#############################################################################################

The json file to dataset shell command is included in repository(json_to_dataset.sh).
To transfer json file to image data, use console cd to the file directory, further run shell command $sh json_to_dataset.sh
