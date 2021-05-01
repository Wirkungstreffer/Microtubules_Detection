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

Data process:
By using Fiji or NIS-Elements Viewer the original ND2 video file can be decomposed into 16bit tiff frame images. Copy the images into file Microtubules_Detection_Master/Data_Process/16bit_img_file
Run  16bit_to_8bit_Transfer.py  to use normalization to convert the frames into 8bit .tiff images.
Run  Convert_tiff_to_png.py  to transform format from tiff to png. This step is for the labeling, and may not be necessary for training and other detection methods.





Semantic segmentation:

Structure and settings introduce:

U-Net convolutional neural network structure

Encode: ResNet34 with initial weights imagenet

Input/Output size:1216x1216

Optimizer: Adam with learning rate 0.001

Loss function: jaccard index

Metrics: iou score

Callbacks: ModelCheckpoint, Tensorboard, EarlyStopping

Batch size: 2(Depend on VRAM)

Epochs: 80(may stop early depend on the patients setting in EarlyStopping)

Evaluation: evaluate the iou scores between images predictions and labels in the validation set.



Pre-training data process:

Randomly separate the labeled data into 2 datasets with the percentile of 10% by running divide_data_for_training_and_testing.py(adjusting the parameter k to set the quantity of test set): training set 90%(333) and test set 10%(37), the test set could be used as a scoring dataset to verify the detection ability and accuracy of all methods (edge detection, region segmentation, semantic segmentation). 

Apply augmentation(First do divide).py of training images set to generate 1500 augmented training data.
Implement reflect padding  to enlarge the images from both datasets from size 1200x1200 to 1216x1216. (integrated in Neural_Network_Training.py)
Divide the augmented training dataset into training set and validation set. (integrated in Neural_Network_Training.py)
The label images from the training set and the validation set are being normalized to make sure the labels are [0, 1]. (integrated in Neural_Network_Training.py)




After-training:

The trained neural network will be acquired as 'MT_1216_Semantic_Segmentation.h5'.

Run Testset_Model_validation.py to get the predict segmentations of test set images. The predictions will be stored in Microtubules_Detection_Master/Semantic_Segmentation/training_data/test_predict_output.

Run MT_Length_Measurement_Testset.py to visualize and measure the length of predicted microtubules, for further digitization of comparison with the test set labels and other detection methods.



#########################################################################

Run Neural_Network_Training_Notebook.ipynb can get all the steps after Pre-training data process step 3.

To transfer json file to image data, use console cd to the file, run $sh json_to_dataset.sh
