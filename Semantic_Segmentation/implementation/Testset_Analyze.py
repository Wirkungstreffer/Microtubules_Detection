
from numpy.lib.function_base import average
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pwlf
import csv
from sklearn import linear_model
from scipy.ndimage import gaussian_filter1d

# Read the csv file   Semantic_Segmentation/implementation/Testset_full_auto/Testset_Only_Labels/200818_xb_reaction2_6um003_testset/Ground_Truth
cnn_data_length_csv = pd.read_csv("Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200820_xl_reaction6_03um001_testset/CNN_Results/Microtubules_Lengths_with_Seed_Concatenation.csv",header=None)
label_data_length_csv = pd.read_csv("Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200820_xl_reaction6_03um001_testset/Ground_Truth/Microtubules_Lengths_with_Seed_Concatenation.csv",header=None)
cnn_MT_label_seed_data_length_csv = pd.read_csv("Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200820_xl_reaction6_03um001_testset/Ground_Truth_Seed_CNN_MT/Microtubules_Lengths_with_Seed_Concatenation.csv",header=None)

error_csv_save_path = 'Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200820_xl_reaction6_03um001_testset/Seeds_Concatenated_Error_List.csv'

# Get data information
cnn_row_number = cnn_data_length_csv.shape[0]
cnn_column_number = cnn_data_length_csv.shape[1]
cnn_data_quantity = cnn_row_number*cnn_column_number

print(cnn_data_length_csv.shape)

print("cnn_row: %d     cnn_column: %d     cnn_total data: %d"%(cnn_row_number,cnn_column_number,cnn_data_quantity))


label_row_number = label_data_length_csv.shape[0]
label_column_number = label_data_length_csv.shape[1]
label_data_quantity = label_row_number*label_column_number

print(label_data_length_csv.shape)

print("label_row: %d     label_column: %d     label_total data: %d"%(label_row_number,label_column_number,label_data_quantity))


cnn_MT_label_seed_row_number = cnn_MT_label_seed_data_length_csv.shape[0]
cnn_MT_label_seed_column_number = cnn_MT_label_seed_data_length_csv.shape[1]
cnn_MT_label_seed_data_quantity = cnn_MT_label_seed_row_number*cnn_MT_label_seed_column_number

print(cnn_MT_label_seed_data_length_csv.shape)

print("cnn_MT_label_seed_row: %d     cnn_MT_label_seed_column: %d     cnn_MT_label_seed_total data: %d"%(cnn_MT_label_seed_row_number,cnn_MT_label_seed_column_number,cnn_MT_label_seed_data_quantity))

#########################################################################################################################


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
non_nan_columns_index_cnn = []

# Keep the non-zero columns
for column_loop_cnn in range(cnn_data_length_csv.shape[1]):
    # Filter out Zero columns
    column_validation_cnn = select_non_nan_column(cnn_data_length_csv[cnn_data_length_csv.columns[column_loop_cnn]])
    
    # Filter out the falsh & miss detected column
    if len(column_validation_cnn) >= 0.4*cnn_data_length_csv.shape[0]:
        non_nan_columns_index_cnn.append(column_loop_cnn+1)

print("cnn alive seeds: ",len(non_nan_columns_index_cnn))

# Create a list to store the non-zero column index
non_nan_columns_index_label = []

# Keep the non-zero columns
for column_loop_label in range(label_data_length_csv.shape[1]):
    # Filter out Zero columns
    column_validation_label = select_non_nan_column(label_data_length_csv[label_data_length_csv.columns[column_loop_label]])
    
    # Filter out the falsh & miss detected column
    if len(column_validation_label) >= 0.4*label_data_length_csv.shape[0]:
        non_nan_columns_index_label.append(column_loop_label+1)

print("label alive seeds: ",len(non_nan_columns_index_label))

#########################################################################################################################

# Create a list to store all cnn data
cnn_total_data_matrix = []

# Save the cnn data column wise
for cnn_column_loop in range(cnn_data_length_csv.shape[1]):
    cnn_column_validation = np.array(list(cnn_data_length_csv[cnn_data_length_csv.columns[cnn_column_loop]]))
    cnn_total_data_matrix.append(cnn_column_validation)


# Create a list to store all label data
label_total_data_matrix = []

# Save the label data column wise
for label_column_loop in range(label_data_length_csv.shape[1]):
    label_column_validation = np.array(list(label_data_length_csv[label_data_length_csv.columns[label_column_loop]]))
    label_total_data_matrix.append(label_column_validation)


# Create a list to store all label data
cnn_MT_label_seed_total_data_matrix = []

# Save the label data column wise
for cnn_MT_label_seed_column_loop in range(cnn_MT_label_seed_data_length_csv.shape[1]):
    cnn_MT_label_seed_column_validation = np.array(list(cnn_MT_label_seed_data_length_csv[cnn_MT_label_seed_data_length_csv.columns[cnn_MT_label_seed_column_loop]]))
    cnn_MT_label_seed_total_data_matrix.append(cnn_MT_label_seed_column_validation)


# Calculate ture positive
TP = 0
for i in range(len(label_total_data_matrix)):
    for j in range(len(label_total_data_matrix[0])):
        if (label_total_data_matrix[i][j] != -1) & (cnn_MT_label_seed_total_data_matrix[i][j] != -1):
            TP = TP +1

print("Total Ture Positive",TP)

# Calculate ture negative
TN = 0
for i in range(len(label_total_data_matrix)):
    for j in range(len(label_total_data_matrix[0])):
        if (label_total_data_matrix[i][j] == -1) & (cnn_MT_label_seed_total_data_matrix[i][j] == -1):
            TN = TN +1

print("Total Ture Negative",TN)

# Calculate false positive
FP = 0
for i in range(len(label_total_data_matrix)):
    for j in range(len(label_total_data_matrix[0])):
        if (label_total_data_matrix[i][j] != -1) & (cnn_MT_label_seed_total_data_matrix[i][j] == -1):
            FP = FP +1

print("Total False Positive",FP)

# Calculate ture negative
FN = 0
for i in range(len(label_total_data_matrix)):
    for j in range(len(label_total_data_matrix[0])):
        if (label_total_data_matrix[i][j] == -1) & (cnn_MT_label_seed_total_data_matrix[i][j] != -1):
            FN = FN +1

print("Total False Negative",FN)


# Calculate accuracy of microtubules detection
#########################################################################################################################

# Create a list to store the non-zero column index
non_nan_columns_index_cnn_MT_label_seed = []

# Keep the non-zero columns
for column_loop_cnn_MT_label_seed in range(cnn_MT_label_seed_data_length_csv.shape[1]):
    # Filter out Zero columns
    column_validation_cnn_MT_label_seed = select_non_nan_column(cnn_MT_label_seed_data_length_csv[cnn_MT_label_seed_data_length_csv.columns[column_loop_cnn_MT_label_seed]])
    
    # Filter out the falsh & miss detected column
    if len(column_validation_cnn_MT_label_seed) >= 0.4*cnn_MT_label_seed_data_length_csv.shape[0]:
        non_nan_columns_index_cnn_MT_label_seed.append(column_loop_cnn_MT_label_seed+1)


#non_nan_columns_index_cnn_MT_label_seed.remove(2)
#non_nan_columns_index_cnn_MT_label_seed.remove(31)
#non_nan_columns_index_cnn_MT_label_seed.remove(42)
#non_nan_columns_index_cnn_MT_label_seed.remove(80)
#non_nan_columns_index_cnn_MT_label_seed.remove(102)

print("Ground truth successful concatenated seeds number list:",non_nan_columns_index_label)
print("CNN detection successful concatenated seeds number list:",non_nan_columns_index_cnn_MT_label_seed)
print("total:",len(non_nan_columns_index_label))
print("total:",len(non_nan_columns_index_cnn_MT_label_seed))

failed_detected_seed_index_list = np.setdiff1d(non_nan_columns_index_label,non_nan_columns_index_cnn_MT_label_seed)
print("The failed concatenated seeds number: ", failed_detected_seed_index_list)


# Calculate error
total_error_list = []

for column_number in non_nan_columns_index_cnn_MT_label_seed:
    
    # Read the non-zero column
    cnn_MT_label_seed_column = cnn_MT_label_seed_data_length_csv[cnn_MT_label_seed_data_length_csv.columns[column_number-1]]
    label_column = label_data_length_csv[label_data_length_csv.columns[column_number-1]]

    error_list = []

    # Mark the TP, TN, FP, FN scenario
    for k in range(len(label_column)):
        if (cnn_MT_label_seed_column[k] != -1) & (label_column[k] != -1):
            error = abs(label_column[k] - cnn_MT_label_seed_column[k])
            error_list.append(error)
        
        if (cnn_MT_label_seed_column[k] == -1) & (label_column[k] != -1):
            error = -1
            error_list.append(error)

        if (cnn_MT_label_seed_column[k] != -1) & (label_column[k] == -1):
            error = -2
            error_list.append(error)

        if (cnn_MT_label_seed_column[k] == -1) & (label_column[k] == -1):
            error = -3
            error_list.append(error)
    
    total_error_list.append(error_list)


# Calculate minimal, maximal, average error value of TP scenario. Count TN, FP, FN quantity
Total_average_list = []
Total_max_list = []

error_save_data = []

for l in range(len(total_error_list)):
    FP_in_TP = 0
    FN_in_TP = 0
    TN_in_TP = 0
    for m in total_error_list[l]:
        if m == -1:
            FP_in_TP += 1
    
    for n in total_error_list[l]:
        if n == -2:
            FN_in_TP += 1

    for o in total_error_list[l]:
        if o == -3:
            TN_in_TP += 1
    
    if (FP_in_TP != 0) & (FN_in_TP == 0) & (TN_in_TP == 0):
        the_errors = [err for err in total_error_list[l] if err >= 0]

        error_save_data.append(the_errors)
        
        if len(the_errors) == 0:
            continue
        else:
            column_error_min = min(the_errors)
            column_error_max = max(the_errors)

        Total_max_list.append(column_error_max)

        column_error_average =average(the_errors)
        Total_average_list.append(column_error_average)
        
        print("NO.%d seed corresponding MT:     min_err %f     max_err %f     avg_err %f     FP %d" %(non_nan_columns_index_cnn_MT_label_seed[l], column_error_min, column_error_max, column_error_average, FP_in_TP))
    
    elif (FP_in_TP != 0) & (FN_in_TP != 0) & (TN_in_TP == 0):
        the_errors = [err for err in total_error_list[l] if err >= 0]
        
        error_save_data.append(the_errors)

        if len(the_errors) == 0:
            continue
        else:
            column_error_min = min(the_errors)
            column_error_max = max(the_errors)

        Total_max_list.append(column_error_max)

        column_error_average =average(the_errors)
        Total_average_list.append(column_error_average)
        
        print("NO.%d seed corresponding MT:     min_err %f     max_err %f     avg_err %f     FP %d     FN %d" %(non_nan_columns_index_cnn_MT_label_seed[l], column_error_min, column_error_max, column_error_average, FP_in_TP, FN_in_TP))
    
    elif (FP_in_TP != 0) & (FN_in_TP == 0) & (TN_in_TP != 0):
        the_errors = [err for err in total_error_list[l] if err >= 0]

        error_save_data.append(the_errors)

        if len(the_errors) == 0:
            continue
        else:
            column_error_min = min(the_errors)
            column_error_max = max(the_errors)
            
        Total_max_list.append(column_error_max)

        column_error_average =average(the_errors)
        Total_average_list.append(column_error_average)
        
        print("NO.%d seed corresponding MT:     min_err %f     max_err %f     avg_err %f     FP %d     TN %d" %(non_nan_columns_index_cnn_MT_label_seed[l], column_error_min, column_error_max, column_error_average,  FP_in_TP, TN_in_TP))
    
    elif (FP_in_TP == 0) & (FN_in_TP != 0) & (TN_in_TP == 0):
        the_errors = [err for err in total_error_list[l] if err >= 0]

        error_save_data.append(the_errors)

        if len(the_errors) == 0:
            continue
        else:
            column_error_min = min(the_errors)
            column_error_max = max(the_errors)
            
        Total_max_list.append(column_error_max)

        column_error_average =average(the_errors)
        Total_average_list.append(column_error_average)
        
        print("NO.%d seed corresponding MT:     min_err %f     max_err %f     avg_err %f     FN %d" %(non_nan_columns_index_cnn_MT_label_seed[l], column_error_min, column_error_max, column_error_average, FN_in_TP))

    elif (FP_in_TP == 0) & (FN_in_TP != 0) & (TN_in_TP != 0):
        the_errors = [err for err in total_error_list[l] if err >= 0]

        error_save_data.append(the_errors)

        if len(the_errors) == 0:
            continue
        else:
            column_error_min = min(the_errors)
            column_error_max = max(the_errors)
            
        Total_max_list.append(column_error_max)

        column_error_average =average(the_errors)
        Total_average_list.append(column_error_average)
        
        print("NO.%d seed corresponding MT:     min_err %f     max_err %f     avg_err %f     FN %d     TN %d" %(non_nan_columns_index_cnn_MT_label_seed[l], column_error_min, column_error_max, column_error_average, FN_in_TP, TN_in_TP))
    
    elif (FP_in_TP == 0) & (FN_in_TP == 0) & (TN_in_TP != 0):
        the_errors = [err for err in total_error_list[l] if err >= 0]

        error_save_data.append(the_errors)

        if len(the_errors) == 0:
            continue
        else:
            column_error_min = min(the_errors)
            column_error_max = max(the_errors)
            
        Total_max_list.append(column_error_max)

        column_error_average =average(the_errors)
        Total_average_list.append(column_error_average)
        
        print("NO.%d seed corresponding MT:     min_err %f     max_err %f     avg_err %f     TN %d" %(non_nan_columns_index_cnn_MT_label_seed[l], column_error_min, column_error_max, column_error_average, TN_in_TP))
    
    elif (FP_in_TP != 0) & (FN_in_TP != 0) & (TN_in_TP != 0):
        the_errors = [err for err in total_error_list[l] if err >= 0]

        error_save_data.append(the_errors)

        if len(the_errors) == 0:
            continue
        else:
            column_error_min = min(the_errors)
            column_error_max = max(the_errors)
            
        Total_max_list.append(column_error_max)

        column_error_average =average(the_errors)
        Total_average_list.append(column_error_average)
        
        print("NO.%d seed corresponding MT:     min_err %f     max_err %f     avg_err %f     FP %d     FN %d     TN %d" %(non_nan_columns_index_cnn_MT_label_seed[l], column_error_min, column_error_max, column_error_average, FP_in_TP, FN_in_TP, TN_in_TP ))

    elif (FP_in_TP == 0) & (FN_in_TP == 0) & (TN_in_TP == 0):
        the_errors = [err for err in total_error_list[l] if err >= 0]

        error_save_data.append(the_errors)

        if len(the_errors) == 0:
            continue
        else:
            column_error_min = min(the_errors)
            column_error_max = max(the_errors)
            
        Total_max_list.append(column_error_max)

        column_error_average =average(the_errors)
        Total_average_list.append(column_error_average)
        
        print("NO.%d seed corresponding MT:     min_err %f     max_err %f     avg_err %f" %(non_nan_columns_index_cnn_MT_label_seed[l], column_error_min, column_error_max, column_error_average))

# Calculate general average, maximal error value

general_average = average(Total_average_list)
general_max = max(Total_max_list)

print("General average error: ", general_average)
print("General maximal error: ", general_max)

# Store the information into csv file
file_csv_1 = open(error_csv_save_path,'w',newline='')
writer_csv_1 = csv.writer(file_csv_1)
for errors in error_save_data:
    writer_csv_1.writerow(errors)