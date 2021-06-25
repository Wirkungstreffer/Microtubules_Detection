import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pwlf
import csv
from sklearn import linear_model
from scipy.ndimage import gaussian_filter1d


#### This script is to read the lengths of microtubules and return microtubules velocities information ####
#### The input is the csv file "implementation/Microtubules_Lengths_with_Seed_Concatenation" ####
#### Output of microtubules rates information will be stored as csv file and some additional images to visualize the linear regressions ####


# Read the csv file
data_length_csv = pd.read_csv("Semantic_Segmentation/implementation/data_output/Microtubules_Lengths_with_Seed_Concatenation.csv", header=None)

# Define a function to select non-zero column
def select_non_nan_column(column):
    # Create a list to store selected column
    column_non_nan = []
    
    # Filter all the zeros
    for non_nan in column:
        if non_nan != -1:
            column_non_nan.append(non_nan)

    return column_non_nan

# Create a list to store the non-zero column index
non_nan_columns_index = []

# Keep the non-zero columns
for column_loop in range(data_length_csv.shape[1]):
    # Filter out Zero columns
    column_validation = select_non_nan_column(data_length_csv[data_length_csv.columns[column_loop]])
    
    # Filter out the falsh & miss detected column
    if len(column_validation) >= 0.4*data_length_csv.shape[0]:
        non_nan_columns_index.append(column_loop) 

# Create a list to store rate information
total_rate_list = []

total_missing_data_list = []
for column_number in non_nan_columns_index:
    
    # Read the non-zero column
    the_column = data_length_csv[data_length_csv.columns[column_number]]
    
    original_x_frame_number = np.array([np.arange(0,len(the_column))])
    original_y_microtubules_length_array = np.array([the_column])

    miss_data_array = np.where(original_y_microtubules_length_array==-1)

    total_missing_data_list.append(miss_data_array[1])

    print("NO.%s_Seed_missing_data"%(column_number+1), miss_data_array[1])

    # Plot the scatter data
    plt.scatter(original_x_frame_number, original_y_microtubules_length_array,marker='.')
    original_image_save_path = "Semantic_Segmentation/implementation/data_output/NO.%s_Seed_Corresponding_Microtubules_Lengths_Scatter_Image.png" %(column_number+1)
    plt.savefig(original_image_save_path)
    plt.clf()
    #plt.show()

    # Define a function that eliminate outliers 
    def reject_outliers(data):
        # Create a list to store filtered data
        data_filtered = []
        data_non_nan = []
        
        # Caculate mean and variance of the data
        for n_z in data:
            if n_z != -1:
                data_non_nan.append(n_z)

        u = np.mean(data_non_nan)
        s = np.std(data_non_nan)

        # Save the data within 2 standard deviation
        for d in data_non_nan:
            if (d>(u-2*s)) & (d<(u+2*s)):
                data_filtered.append(d)
        
        return data_filtered

    # Delete outliers
    Case_Microtubules_Delete_Outliers = reject_outliers(the_column)

    #Case_Microtubules_Delete_Outliers = gaussian_filter1d(Case_Microtubules_Delete_Outliers,3)

    # Transfer data into array for further process
    x_frame_number = np.array([np.arange(0,len(Case_Microtubules_Delete_Outliers))])
    y_microtubules_length_array = np.array([Case_Microtubules_Delete_Outliers])

    length_array = y_microtubules_length_array[0]
    #length_array = gaussian_filter1d(y_microtubules_length_array[0],6)

    # Keep the list form of original length data
    x_frame_number_array = np.array([np.arange(0,len(Case_Microtubules_Delete_Outliers))])
    y_microtubules_length = Case_Microtubules_Delete_Outliers

    # Define the local with neighbor number
    local_number = int(len(y_microtubules_length)*0.1)

    # Create local extreme index list
    minimal_index = []
    maximal_index = []

    # Count the local extreme number
    minimal_counter = 0
    maximal_counter = 0

    # Find the local peaks & trough
    for extreme_index in range(local_number, len(length_array)-local_number):

        # Create a list to store local neighbor
        local_list = []

        # Save local neighbot length as list
        for local_loop in range(1, local_number):
            previous_length_number = length_array[extreme_index-local_loop]
            local_list.append(previous_length_number)
            followed_length_number = length_array[extreme_index+local_loop]
            local_list.append(followed_length_number)

        # If the length is the largest among all neighbors, it's local maximal
        if length_array[extreme_index] >= max(local_list) :
            maximal_index.append(extreme_index)
            maximal_counter = maximal_counter + 1

        # If the length is the shortest among all neighbors, it's local minimal
        if length_array[extreme_index] <= min(local_list):
            minimal_index.append(extreme_index)
            minimal_counter = minimal_counter + 1

    # Get the local extreme length information
    local_extreme_max_length = []
    local_extreme_min_length = []

    for max_length in maximal_index:
        local_extreme_max_length.append(length_array[max_length])

    for min_length in minimal_index:
        local_extreme_min_length.append(length_array[min_length])

    # Delete the equal length to get correct local extreme counter number
    equal_max_length = len(local_extreme_max_length) - len(set(local_extreme_max_length))
    equal_min_length = len(local_extreme_min_length) - len(set(local_extreme_min_length))

    equal_length_number = equal_max_length + equal_min_length

    # Create local maximal mask for further visulizaiton & process
    extreme_max_mask = []

    for extreme_max_log in range(len(length_array)):
        if any(extreme_max_log == index for index in maximal_index):
            extreme_max_mask.append(True)
        else :
            extreme_max_mask.append(False)

    # Create local minimal mask for further visulizaiton & process
    extreme_min_mask = []

    for extreme_min_log in range(len(length_array)):
        if any(extreme_min_log == index for index in minimal_index):
            extreme_min_mask.append(True)
        else :
            extreme_min_mask.append(False)

    # Get the extreme value mask
    extreme_mask = []

    for extreme_log in range(len(length_array)):
        extreme_mask.append(extreme_max_mask[extreme_log] or extreme_min_mask[extreme_log])

    # Get the normal nonextreme value mask
    nonextreme_mask = np.logical_not(extreme_mask)

    # Add up totall local extreme value quantity 
    total_local_extreme_number = minimal_counter + maximal_counter - equal_length_number

    print("The total local extreme quantity of NO.%d seed corresponding microtubule is: "%(column_number+1),total_local_extreme_number)

    # Set the quantity of segmentations of linear regression piece
    breakpoint_number = total_local_extreme_number  + 1

    # If the data is monotonically increasing, use random sample consensus to get the slope
    if breakpoint_number == 1:
        
        # Reshape data to the algorithm input format
        X = x_frame_number.reshape(-1, 1) 
        y = y_microtubules_length_array.reshape(-1, 1)

        # Robustly fit linear model with RANSAC algorithm
        ransac = linear_model.RANSACRegressor()
        ransac.fit(X, y)
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)

        # Predict data of estimated models
        line_X = np.arange(X.min(), X.max())[:, np.newaxis]
        line_y_ransac = ransac.predict(line_X)

        # Get the slope
        slopes = ransac.estimator_.coef_[0]

        # Draw and save the piecewise linear regression image
        plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.', label='Inliers')
        plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.', label='Outliers')
        plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=2,label='RANSAC regressor')
        plt.legend(loc='lower right')
        plt.xlabel("Frame")
        plt.ylabel("Microtubules Length")
        plt.title("NO.%s Seed Corresponding Microtubules Lengths Linear Regressioin"%(column_number+1))
        pwlf_image_save_path = "Semantic_Segmentation/implementation/data_output/NO.%s_Seed_Corresponding_Microtubules_Lengths_Linear_Regressioin.png" %(column_number+1)
        plt.savefig(pwlf_image_save_path)
        plt.clf()

    elif breakpoint_number > 1: 
        # Fit in the data
        my_pwlf = pwlf.PiecewiseLinFit(x_frame_number_array, y_microtubules_length)
        breaks = my_pwlf.fit(breakpoint_number)

        # Give the different linear regression breakpoints information
        breaks_int = []
        for bp_number in breaks:
            breaks_int.append(round(bp_number))

        # Get the first derivative of the linear regressions
        slopes = my_pwlf.calc_slopes()

        # Make the linear regression prediction
        x_frame_number_hat = np.linspace(x_frame_number.min(), x_frame_number.max(), 10000)
        y_microtubules_length_hat = my_pwlf.predict(x_frame_number_hat)

        # Draw and save the piecewise linear regression image
        #plt.plot(x_frame_number, y_microtubules_length_array, markersize = 2, marker = 'o',color='gold')
        plt.scatter(x_frame_number.reshape(-1, 1)[nonextreme_mask], y_microtubules_length_array.reshape(-1, 1)[nonextreme_mask], color='blue', marker='.', label='Nonextreme')
        plt.scatter(x_frame_number.reshape(-1, 1)[extreme_min_mask], y_microtubules_length_array.reshape(-1, 1)[extreme_min_mask], color='red', marker='.', label='Local Minimal')
        plt.scatter(x_frame_number.reshape(-1, 1)[extreme_max_mask], y_microtubules_length_array.reshape(-1, 1)[extreme_max_mask], color='gold', marker='.', label='Local Maximal')
        plt.plot(x_frame_number_hat, y_microtubules_length_hat, '-')
        plt.legend(loc='lower right')
        plt.xlabel("Frame")
        plt.ylabel("Microtubules Length")
        plt.title("NO.%s Seed Corresponding Microtubules Lengths Linear Regressioin"%(column_number+1))
        pwlf_image_save_path = "Semantic_Segmentation/implementation/data_output/NO.%s_Seed_Corresponding_Microtubules_Lengths_Linear_Regressioin.png" %(column_number+1)
        plt.savefig(pwlf_image_save_path)
        plt.clf()

    print("The breakpoint of NO.%d seed corresponding microtubule is:"%(column_number+1),breakpoint_number)

    # Use the scale proportion to get the rate
    frame_second_proportion = 5        # 5 sec per frame
    length_pixel_proportion = 9.1287   # 9.1287 pixel per uM

    # Store the rate
    rate_list = []
    for slope in slopes:
        rate = slope*(1/length_pixel_proportion)/frame_second_proportion
        rate_list.append(rate)

    print("The rates of NO.%d seed corresponding microtubule: "%(column_number+1),rate_list)

    total_rate_list.append(rate_list)


# Define a sublist expand function to expand the zipped data
def expand(lst):
    try:
        for a in lst:
            for b in expand(a):
                yield b
    except TypeError:
        yield lst

# Column index and seed index have the differ of 1 
index_correspond_number = []
for index in non_nan_columns_index:
    index_correspond_number.append(index + 1)

# Zip the column index with corresponding slope/rate information
rate_information = list(zip(index_correspond_number, total_rate_list))

# Expand the zipped data
rate_list_prepare_csv = []
for info in range(len(rate_information)):
    rate_list_prepare_csv.append(expand(rate_information[info]))

# Store the rates information in to csv file
rate_list_file_csv = open('Semantic_Segmentation/implementation/data_output/Microtubules_Rate_List.csv','w',newline='')
rate_list_writer_csv = csv.writer(rate_list_file_csv)
for row in rate_list_prepare_csv:
    rate_list_writer_csv.writerow(row)