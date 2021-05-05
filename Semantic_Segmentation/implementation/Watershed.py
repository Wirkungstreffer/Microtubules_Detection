import os
os.environ['DISPLAY'] = ':1'
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure, color, io

img = cv2.imread("Semantic_Segmentation/implementation/prediction_seed/seed_predict_0.png")  #Read as color (3 channels)
img_grey = img[:,:,0]

## transform the unet result to binary image
#Threshold image to binary using OTSU. ALl thresholded pixels will be set to 255
ret1, thresh = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


# Morphological operations to remove small noise - opening
#To remove holes we can use closing
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

#from skimage.segmentation import clear_border
#opening = clear_border(opening) #Remove edge touching grains. 
#Check the total regions found before and after applying this. 


sure_bg = cv2.dilate(opening,kernel,iterations=2)


dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)


#Let us threshold the dist transform by starting at 1/2 its max value.
ret2, sure_fg = cv2.threshold(dist_transform, 0.2*dist_transform.max(),255,0)

sure_fg = np.uint8(sure_fg)
separate_parts = cv2.subtract(opening, sure_fg)

fig, ax = plt.subplots(ncols=3, figsize=(20, 20))
ax[0].imshow(dist_transform)
ax[0].set_title('dist_transform')
ax[0].axis('off')

ax[1].imshow(sure_fg)
ax[1].set_title('sure_fg')
ax[1].axis('off')

ax[2].imshow(separate_parts)
ax[2].set_title('separate_parts')
ax[2].axis('off')

plt.show()

# Unknown ambiguous region is nothing but bkground - foreground
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Unknown regions will be labeled 0. 
#For markers let us use ConnectedComponents. 
ret3, markers = cv2.connectedComponents(sure_fg)

#One problem rightnow is that the entire background pixels is given value 0.
#This means watershed considers this region as unknown.
#So let us add 10 to all labels so that sure background is not 0, but 10
markers = markers+10

# Now, mark the region of unknown with zero
markers[unknown==255] = 0
#plt.imshow(markers, cmap='gray')   #Look at the 3 distinct regions.

#Now we are ready for watershed filling. 
markers = cv2.watershed(img, markers)
#plt.imshow(markers, cmap='gray')

#Let us color boundaries in yellow. 
img[markers == -1] = [0,255,255]  

img2 = color.label2rgb(markers, bg_label=0)


#fig, ax = plt.subplots(ncols=2, figsize=(20, 20))
#ax[0].imshow(img)
#ax[0].set_title('Overlay on original image')
#ax[0].axis('off')

#ax[1].imshow(img2)
#ax[1].set_title('Colored Grains')
#ax[1].axis('off')

#plt.show()

#cv2.imshow('Overlay on original image', img)
#cv2.imshow('Colored Grains', img2)
#cv2.waitKey(0)