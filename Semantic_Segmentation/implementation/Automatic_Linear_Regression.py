from numpy.core.fromnumeric import mean
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


# Define local percentage of total frames number in local extrame value detection if adjustment is needed
local_percentage_of_total_frames_number = 0.1

# The scale proportion
frame_second_proportion = 5        # 5 sec per frame
length_pixel_proportion = 9.1287   # 9.1287 pixel per uM

# Read the csv file
data_length_csv = pd.read_csv("Semantic_Segmentation/implementation/data_output/Microtubules_Lengths_with_Seed_Concatenation.csv", header=None)

# Define a function to select non-negative column
def select_non_nan_column(column):
    # Create a list to store selected column
    column_non_nan = []
    
    # Filter all the negative
    for non_nan in column:
        if non_nan != -1:
            column_non_nan.append(non_nan)

    return column_non_nan

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

# Create a list to store the non-zero column index
non_nan_columns_index = []

# Keep the non-negative columns
for column_loop in range(data_length_csv.shape[1]):
    # Filter out negative columns
    column_validation = select_non_nan_column(data_length_csv[data_length_csv.columns[column_loop]])
    
    # Filter out the falsh & miss detected column
    if len(column_validation) >= 0.2*data_length_csv.shape[0]:
        non_nan_columns_index.append(column_loop) 

# Delete identical data
identical_pair_list = []

for columns in range(len(non_nan_columns_index)):
    
    # Read columns
    columns_data = data_length_csv[data_length_csv.columns[non_nan_columns_index[columns]]]
    
    # Get data mean value and variance value
    mean_value = np.mean(columns_data)
    variance_value = np.var(columns_data)

    # Compare with other columns
    for other_columns in range(columns + 1, len(non_nan_columns_index)):
        # Read other coulumns
        other_columns_data = data_length_csv[data_length_csv.columns[non_nan_columns_index[other_columns]]]
        
        # Get other data mean value and variance value
        other_mean_value = np.mean(other_columns_data)
        other_vairance_value = np.var(other_columns_data)

        # Calculate difference percentage
        differ_mean = (np.abs(mean_value - other_mean_value))/(mean_value)
        differ_variance = (np.abs(variance_value - other_vairance_value))/(variance_value)

        # If the difference is too small, consider two columns are the same
        if (differ_mean < 0.05) & (differ_variance < 0.05):
            identical_pair_list.append(columns)
            
# Remove the identical column
for identical_index in identical_pair_list:
    non_nan_columns_index.remove(non_nan_columns_index[identical_index])

# Create a list to store rate information
total_rate_list = []
total_positive_rate_list = []
total_negative_rate_list = []

# Create a list to store event time information
total_event_time_list = []
total_positive_event_time_list = []
total_negative_event_time_list = []

# Create a list to store event length information
total_event_length_list = []
total_positive_event_length_list = []
total_negative_event_length_list = []

# Create a list to store missed data number
total_missing_data_list = []

# Start the linear regression calculation
for column_number in non_nan_columns_index:
    
    # Read the non-zero column
    the_column = data_length_csv[data_length_csv.columns[column_number]]
    
    original_x_frame_number = np.array([np.arange(0,len(the_column))])
    original_y_microtubules_length_array = np.array([the_column])

    miss_data_array = np.where(original_y_microtubules_length_array==-1)

    total_missing_data_list.append(miss_data_array[1])

    #print("NO.%s_Seed_missing_data"%(column_number+1), miss_data_array[1])

    # Plot the scatter data
    plt.scatter(original_x_frame_number, original_y_microtubules_length_array,marker='.')
    plt.xlabel("Frame")
    plt.ylabel("Microtubules Length")
    plt.title("NO.%s Seed Corresponding Microtubules Lengths scatter plot"%(column_number+1))
    original_image_save_path = "Semantic_Segmentation/implementation/data_output/NO.%s_Seed_Corresponding_Microtubules_Lengths_Scatter_Image.png" %(column_number+1)
    plt.savefig(original_image_save_path)
    plt.clf()
    #plt.show()

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
    local_number = int(len(y_microtubules_length)*local_percentage_of_total_frames_number)

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

        # Calculate duration that the event lasted
        event_frame_length = max(X) - min(X)
        
        # Get the slope
        slopes = ransac.estimator_.coef_[0]

        # Multiply event frame length and corresponding slope to get event length
        event_length = []
        for the_event in range(len(event_frame_length)):
            the_event_length = (event_frame_length[the_event])*(slopes[the_event])
            event_length.append(the_event_length)

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
        print("original breakpoint number",breakpoint_number)
        # Make sure the adjacent rates are positive and negative opposite
        positive_negative_opposite_judgment = 0
        judge_breakpoint_number = breakpoint_number

        while (positive_negative_opposite_judgment == 1)|(judge_breakpoint_number == 2):
            # Calculate all slopes in current breakpoint numbers setting
            judge_pwlf = pwlf.PiecewiseLinFit(x_frame_number_array, y_microtubules_length)
            judge_break = judge_pwlf.fit(judge_breakpoint_number)
            judge_slopes = judge_pwlf.calc_slopes()
            judge_list = []
            
            # Judge if adjacent rates are positive and negative opposite
            for the_judge_slope in range(len(judge_slopes)-1):
                previous_slope = judge_slopes[the_judge_slope]
                later_slope = judge_slopes[the_judge_slope]
                if previous_slope*later_slope < 0:
                    judge_list.append(-1)
                elif previous_slope*later_slope > 0:
                    judge_list.append(1)
            
            # If all adjacent rates are positive and negative opposite break the loop, otherwise breakpoints number minus one
            if all(s < 0 for s in judge_list):
                positive_negative_opposite_judgment = 1
            else:
                positive_negative_opposite_judgment = 0
                judge_breakpoint_number = judge_breakpoint_number - 1

        
        print("judge breakpoint number",judge_breakpoint_number)

        # Fit in the data
        my_pwlf = pwlf.PiecewiseLinFit(x_frame_number_array, y_microtubules_length)
        breaks = my_pwlf.fit(breakpoint_number)

        # Give the different linear regression breakpoints information
        breaks_int = []
        for bp_number in breaks:
            breaks_int.append(round(bp_number))
        
        # Subtract adjacent breakpoint to get the event frame length
        event_frame_length = []
        for event in range(len(breaks_int) - 1):
            frame_length = breaks_int[event + 1] - breaks_int[event]
            event_frame_length.append(frame_length)

        # Get the first derivative of the linear regressions
        slopes = my_pwlf.calc_slopes()

        # Multiply event frame length and corresponding slope to get event length
        event_length = []
        for the_event in range(len(event_frame_length)):
            the_event_length = (event_frame_length[the_event])*(slopes[the_event])
            event_length.append(the_event_length)

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

    #print("The breakpoint of NO.%d seed corresponding microtubule is:"%(column_number+1),breakpoint_number)

    # Create lists to store information
    rate_list = []
    positive_rate_list = []
    negative_rate_list = []
    
    event_time_list = []
    positive_event_time_list = []
    negative_event_time_list = []

    event_length_list = []
    positive_event_length_list = []
    negative_event_length_list = []

    for slope_index in range(len(slopes)):
        # Store the rate
        rate = slopes[slope_index]*(1/length_pixel_proportion)/frame_second_proportion
        rate_list.append(rate)
        
        # Store the event duration
        time_real = event_frame_length[slope_index]*frame_second_proportion
        event_time_list.append(time_real)

        # Store the event length
        length_real = np.abs(event_length[slope_index])/length_pixel_proportion
        event_length_list.append(length_real)

        # Store the event information depend on the rate positive or negative character
        if rate > 0:
            positive_rate_list.append(rate)
            positive_event_time_list.append(time_real)
            positive_event_length_list.append(length_real)
        elif rate < 0:
            negative_rate_list.append(rate)
            negative_event_time_list.append(time_real)
            negative_event_length_list.append(length_real)


    print("The rates of NO.%d seed corresponding microtubule: "%(column_number+1),rate_list)

    total_rate_list.append(rate_list)
    total_positive_rate_list.append(positive_rate_list)
    total_negative_rate_list.append(negative_rate_list)

    total_event_time_list.append(event_time_list)
    total_positive_event_time_list.append(positive_event_time_list)
    total_negative_event_time_list.append(negative_event_time_list)

    total_event_length_list.append(event_length_list)
    total_positive_event_length_list.append(positive_event_length_list)
    total_negative_event_length_list.append(negative_event_length_list)

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

# Total seeds number and seeds generated microtubules
total_seeds_number_and_microtubules = "Total seeds number:%d, seeds generate microtubules number:%d" %(data_length_csv.shape[1],len(non_nan_columns_index))

# Zip the column index with corresponding slope/rate information
rate_information = list(zip(index_correspond_number, total_rate_list))
positive_rate_information = list(zip(index_correspond_number, total_positive_rate_list))
negative_rate_information = list(zip(index_correspond_number, total_negative_rate_list))

# Zip the column index with corresponding event duration information
event_time_information = list(zip(index_correspond_number, total_event_time_list))
positive_event_time_information = list(zip(index_correspond_number, total_positive_event_time_list))
negative_event_time_information = list(zip(index_correspond_number, total_negative_event_time_list))

# Zip the column index with corresponding event length information
event_length_information = list(zip(index_correspond_number, total_event_length_list))
positive_event_length_information = list(zip(index_correspond_number, total_positive_event_length_list))
negative_event_length_information = list(zip(index_correspond_number, total_negative_event_length_list))

# Expand the zipped rate and events data
rate_event_list_prepare_csv = []
positive_rate_event_list_prepare_csv = []
negative_rate_event_list_prepare_csv = []

# Add seeds and microtubules number information into csv data
total_seeds_number_and_microtubules = [total_seeds_number_and_microtubules]
rate_event_list_prepare_csv.append(total_seeds_number_and_microtubules)
positive_rate_event_list_prepare_csv.append(total_seeds_number_and_microtubules)
negative_rate_event_list_prepare_csv.append(total_seeds_number_and_microtubules)

# Store inforamtion into csv lists
for info in range(len(rate_information)):
    rate_event_list_prepare_csv.append(expand(rate_information[info]))
    rate_event_list_prepare_csv.append(expand(event_time_information[info]))
    rate_event_list_prepare_csv.append(expand(event_length_information[info]))

for positive_info in range(len(positive_rate_information)):
    positive_rate_event_list_prepare_csv.append(expand(positive_rate_information[positive_info]))
    positive_rate_event_list_prepare_csv.append(expand(positive_event_time_information[positive_info]))
    positive_rate_event_list_prepare_csv.append(expand(positive_event_length_information[positive_info]))

for negative_info in range(len(negative_rate_information)):
    negative_rate_event_list_prepare_csv.append(expand(negative_rate_information[negative_info]))
    negative_rate_event_list_prepare_csv.append(expand(negative_event_time_information[negative_info]))
    negative_rate_event_list_prepare_csv.append(expand(negative_event_length_information[negative_info]))


# Store the rates and events information in to csv file
rate_list_file_csv = open('Semantic_Segmentation/implementation/data_output/Microtubules_Rate_Event_List.csv','w',newline='')
rate_list_writer_csv = csv.writer(rate_list_file_csv)
for row in rate_event_list_prepare_csv:
    rate_list_writer_csv.writerow(row)

# Store the positive rates and events information in to csv file 
positive_rate_list_file_csv = open('Semantic_Segmentation/implementation/data_output/Microtubules_Positive_Rate_Event_List.csv','w',newline='')
positive_rate_list_writer_csv = csv.writer(positive_rate_list_file_csv)
for positive_row in positive_rate_event_list_prepare_csv:
    positive_rate_list_writer_csv.writerow(positive_row)

# Store the negative rates and events information in to csv file 
negative_rate_list_file_csv = open('Semantic_Segmentation/implementation/data_output/Microtubules_Negative_Rate_Event_List.csv','w',newline='')
negative_rate_list_writer_csv = csv.writer(negative_rate_list_file_csv)
for negative_row in negative_rate_event_list_prepare_csv:
    negative_rate_list_writer_csv.writerow(negative_row)