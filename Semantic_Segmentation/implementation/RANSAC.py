import pandas as pd


df = pd.read_csv("Semantic_Segmentation/implementation/Microtubules_Lengths_with_Seed_Concatenation.csv")
first_column = df[df.columns[12]]
#print(first_column)


import numpy as np
from sklearn.preprocessing import PolynomialFeatures
# Define a function that eliminate outliers 
def reject_outliers(data):
    # Create a list to store filtered data
    data_filtered = []
    data_non_zero = []
    
    # Caculate mean and variance of the data
    for n_z in data:
        if n_z != 0:
            data_non_zero.append(n_z)

    u = np.mean(data_non_zero)
    s = np.std(data_non_zero)

    # Save the data within 2 standard deviation
    for d in data_non_zero:
        if (d>(u-2*s)) & (d<(u+2*s)):
            data_filtered.append(d)
    
    return data_filtered


import numpy as np
from matplotlib import pyplot as plt

from sklearn import linear_model, datasets


# Delete outliers
Case_Microtubules_Delete_Outliers = reject_outliers(first_column)

# Scatter plot the length
x_frame_number = np.array([np.arange(0,len(Case_Microtubules_Delete_Outliers))])
y_microtubules_length_array = np.array([Case_Microtubules_Delete_Outliers])

y_input = []

for length in y_microtubules_length_array:
    y_input.append([length])


X = x_frame_number.reshape(-1, 1) 

y = y_microtubules_length_array.reshape(-1, 1)


# Robustly fit linear model with RANSAC algorithm
ransac = linear_model.RANSACRegressor(residual_threshold = 4)
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
print(inlier_mask)
outlier_mask = np.logical_not(inlier_mask)
#print(ransac.score(X, y))

poly = PolynomialFeatures(degree=2)
poly.fit_transform(X,y)

# Predict data of estimated models
line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_y_ransac = ransac.predict(line_X)
slope = ransac.estimator_.coef_[0]
slope = slope/9.1287/5
frame_second_proportion = 5        # 5 sec per pixel
length_pixel_proportion = 9.1287
#print('Slope:%f'%(slope))

lw =2

plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.', label='Outliers')
plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw,label='RANSAC regressor')
plt.legend(loc='lower right')
plt.xlabel("Input")
plt.ylabel("Response")
plt.savefig("Semantic_Segmentation/implementation/RANSAC.png")
plt.show()


import cv2
import os
# If cv2.imshow() function dosen't work, use the followed line of code
os.environ['DISPLAY'] = ':1'

implementation_img = cv2.imread("Semantic_Segmentation/implementation/input_image/200818_Xb_Reaction2_6uM003_001.png", cv2.IMREAD_COLOR)

image_size_x = implementation_img.shape[0]
image_size_y = implementation_img.shape[1]

compensate_x = int(image_size_x/32+1)*32 - image_size_x
compensate_y = int(image_size_y/32+1)*32 - image_size_y

print(compensate_x)

implementation_reflect_img = cv2.copyMakeBorder(implementation_img,0,compensate_x,0,compensate_y,cv2.BORDER_REFLECT)

print(implementation_reflect_img.shape)
#cv2.imshow("implement", implementation_reflect_img)
#cv2.waitKey(0)
implementation_prediction_image_cropped = implementation_reflect_img[0:image_size_x, 0:image_size_y]
cv2.imshow("implement", implementation_prediction_image_cropped)
cv2.waitKey(0)
print(implementation_prediction_image_cropped.shape)
