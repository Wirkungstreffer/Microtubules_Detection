import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pwlf
import csv
from itertools import islice




#
column_number = 14
correct_breakpoint_number = 7

# Read the csv file
data_length_csv = pd.read_csv("Semantic_Segmentation/implementation/Microtubules_Lengths_with_Seed_Concatenation.csv")

the_column = data_length_csv[data_length_csv.columns[column_number-1]]

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

# Delete outliers
Case_Microtubules_Delete_Outliers = reject_outliers(the_column)

# Scatter plot the length
x_frame_number = np.array([np.arange(0,len(Case_Microtubules_Delete_Outliers))])
y_microtubules_length_array = np.array([Case_Microtubules_Delete_Outliers])

length_array = y_microtubules_length_array[0]

# Transfer into array for further process
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

print("The total local extreme quantity of NO.%d is: "%(column_number),total_local_extreme_number)

# Set the quantity of segmentations of linear regression piece
breakpoint_number = total_local_extreme_number  + 1

breakpoint_number = correct_breakpoint_number

print("The breakpoint of NO.%d is:"%(column_number),breakpoint_number)

# Fit in the data
my_pwlf = pwlf.PiecewiseLinFit(x_frame_number_array, y_microtubules_length)
breaks = my_pwlf.fit(breakpoint_number)

# Give the different linear regression breakpoints information
breaks_int = []
for bp_number in breaks:
    breaks_int.append(round(bp_number))

# Get the first derivative of the linear regressions
slopes = my_pwlf.calc_slopes()
#print("The orginal slopes: ",slopes)

# Use the scale proportion to get the rate
frame_second_proportion = 5        # 5 sec per frame
length_pixel_proportion = 9.1287   # 9.1287 pixel per uM

# Store the rate
rate_list = []
for slope in slopes:
    rate = slope*(1/length_pixel_proportion)/frame_second_proportion
    rate_list.append(rate)

print("The rates of NO.%d: "%(column_number),rate_list)


# Zip the column index with corresponding slope/rate information
rate_information = list([column_number]) + list(rate_list)

# Store the rates information in to csv file
#rate_list_file_csv = open('Semantic_Segmentation/implementation/Microtubules_Rate_List.csv','w',newline='')
#rate_list_writer_csv = csv.writer(rate_list_file_csv)
#rate_list_writer_csv.writerow(rate_information)

#with open('Semantic_Segmentation/implementation/Microtubules_Rate_List.csv','a') as data_frame:
#    writer = csv.writer(data_frame)
#    writer.writerow(rate_information)

# Read the csv file
first_column = []    
rate_csv = open ('Semantic_Segmentation/implementation/Microtubules_Rate_List.csv','r')
rate_data = csv.reader(rate_csv)
for column in rate_data:
    if column:
        first_column.append(int(column[0]))

print(first_column)

list2 = []
with open('Semantic_Segmentation/implementation/Microtubules_Rate_List.csv') as f:
    for row in f:
        list2.append(row.split()[0])

print(list2)

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
plt.title("Number_%s_Microtubules_Lengths_Linear_Regressioin"%(column_number))
pwlf_image_save_path = "Semantic_Segmentation/implementation/Number_%s_Microtubules_Lengths_Linear_Regressioin" %(column_number)
plt.savefig(pwlf_image_save_path)
plt.clf()





# The following code is to separate small events and calculate slopes if needed
#########################################################################################################################

# Calculate the intervals between breakpoints
breaks_length = []
for b in range(len(breaks_int)-1):
    the_length = breaks_int[b+1] - breaks_int[b]
    breaks_length.append(the_length)

#print("The events length: ",breaks_length)

# Separate the data according to the breakpoints intervals
def unequal_divide(iterable, chunks):
    it = iter(iterable)
    return [list(islice(it, c)) for c in chunks]

y_events_microtubules_length = unequal_divide(y_microtubules_length, breaks_length)


# Use piecewise linear regression in small events
def separate_small_event(small_event_separate_number):

    # Select small event
    small_event_separate = small_event_separate_number

    # Make a x-axis number array
    x_small_event_frame_number_array = np.array([np.arange(0, breaks_length[small_event_separate])])
    y_small_event_microtubules_length = y_events_microtubules_length[small_event_separate]

    # Define how many breakpoint in the small event
    small_event_breakpoint_number = 3

    # Fit the small events data into piecewise linear regression function
    small_event_my_pwlf = pwlf.PiecewiseLinFit(x_small_event_frame_number_array, y_small_event_microtubules_length)
    small_event_breaks = small_event_my_pwlf.fit(small_event_breakpoint_number)

    # Get the slope
    small_event_slopes = small_event_my_pwlf.calc_slopes()
    #print("The original slopes in small event: ",small_event_slopes)

    # Use the scale proportion to get the rate
    frame_second_proportion = 5        # 5 sec per pixel
    length_pixel_proportion = 9.1287   # 9.1287 pixel per uM

    # Store the rate
    small_event_rate_list = []
    for slope in small_event_slopes:
        rate = slope*(1/length_pixel_proportion)/frame_second_proportion
        small_event_rate_list.append(rate)

    print("The rates in small event: ",small_event_rate_list)

    #rate_list_file_csv = open('Semantic_Segmentation/implementation/Number_%s_Microtubules_Number_%d_Small_Event_Rate_List.csv'%(column_number , small_event_separate),'w',newline='')
    #rate_list_writer_csv = csv.writer(rate_list_file_csv)
    #rate_list_writer_csv.writerow(small_event_rate_list)

    # Make the small event prediction
    x_small_event_frame_number_hat = np.linspace(x_small_event_frame_number_array.min(), x_small_event_frame_number_array.max(), 10000)
    y_small_event_microtubules_length_hat = small_event_my_pwlf.predict(x_small_event_frame_number_hat)

    # Draw and save the small event piecewise linear regression image
    plt.figure()
    plt.xlabel("Frame in small event")
    plt.ylabel("Microtubules Length in small event")
    plt.title("Number_%s_Microtubules_Number_%d_Small_Event_Lengths_Linear_Regressioin" %(column_number , small_event_separate))
    plt.plot(x_small_event_frame_number_array, np.array([y_small_event_microtubules_length]), markersize = 2, marker = 'o')
    plt.plot(x_small_event_frame_number_hat, y_small_event_microtubules_length_hat, '-')
    small_event_pwlf_image_save_path = "Semantic_Segmentation/implementation/Number_%s_Microtubules_Number_%d_Small_Event_Lengths_Linear_Regressioin" %(column_number , small_event_separate)
    plt.savefig(small_event_pwlf_image_save_path)

# Run the small event separation if necessary
#separate_small_event(2)