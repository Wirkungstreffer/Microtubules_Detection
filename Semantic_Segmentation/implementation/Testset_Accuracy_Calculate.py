import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv


#########################################################################################################################

Error_List_csv_1 = pd.read_csv("Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200818_xb_reaction2_6um003_testset/Error_List.csv",header=None)
Seeds_Concatenated_Error_List_csv_1 = pd.read_csv("Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200818_xb_reaction2_6um003_testset/Seeds_Concatenated_Error_List.csv",header=None)
Error_Histogramm_path_1 = "Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200818_xb_reaction2_6um003_testset/Error_Histogramm.png"
Error_Density_Histogramm_path_1 = "Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200818_xb_reaction2_6um003_testset/Error_Density_Histogramm.png"
Seeds_Concatenated_Error_Histogramm_path_1 = "Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200818_xb_reaction2_6um003_testset/Seeds_Concatenated_Error_Histogramm.png"
Seeds_Concatenated_Error_Density_Histogramm_path_1 = "Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200818_xb_reaction2_6um003_testset/Seeds_Concatenated_Error_Density_Histogramm.png"

Error_List_csv_1_non_nan = [x for x in Error_List_csv_1.values.flatten() if str(x) != 'nan']
Seeds_Concatenated_Error_List_csv_1_non_nan = [x for x in Seeds_Concatenated_Error_List_csv_1.values.flatten() if str(x) != 'nan']

Error_List_csv_1_larger_error = [x for x in Error_List_csv_1.values.flatten() if x > 10 ]
Seeds_Concatenated_Error_List_csv_1_larger_error = [x for x in Seeds_Concatenated_Error_List_csv_1.values.flatten() if x > 10]

error_average_1 = np.mean(Error_List_csv_1_non_nan)
error_min_1 = min(Error_List_csv_1_non_nan)
error_max_1 = max(Error_List_csv_1_non_nan)
error_var_1 = np.var(Error_List_csv_1_non_nan)

seed_concatenated_error_average_1 = np.mean(Seeds_Concatenated_Error_List_csv_1_non_nan)
seed_concatenated_error_min_1 = min(Seeds_Concatenated_Error_List_csv_1_non_nan)
seed_concatenated_error_max_1 = max(Seeds_Concatenated_Error_List_csv_1_non_nan)
seed_concatenated_error_var_1 = np.var(Seeds_Concatenated_Error_List_csv_1_non_nan)

print("1.Total data quantity",len(Error_List_csv_1_non_nan))
print("1.The error average: %f     1.The minimal error: %f     1.The maximal error: %f     1.The error variance: %f"%(error_average_1, error_min_1, error_max_1, error_var_1))
print("1.The error larger than 10 pixels quantity: ",len(Error_List_csv_1_larger_error))

print("1.Total seed concatenated error data quantity",len(Seeds_Concatenated_Error_List_csv_1.values.flatten()))
print("1.The seed concatenated error average: %f     1.The seed concatenated error minimal error: %f     1.The seed concatenated error maximal error: %f     1.The seed concatenated error variance: %f"%(seed_concatenated_error_average_1, seed_concatenated_error_min_1, seed_concatenated_error_max_1, seed_concatenated_error_var_1))
print("1.The seed concatenated error larger than 10 pixels quantity: ",len(Seeds_Concatenated_Error_List_csv_1_larger_error))
print('\n')

plt.hist(Error_List_csv_1_non_nan, density=False, bins=80)  # density=False would make counts
plt.ylabel('Frequence')
plt.xlabel('Error')
plt.savefig(Error_Histogramm_path_1)
plt.clf()

plt.hist(Error_List_csv_1_non_nan, density=True, bins=80)  # density=False would make counts
plt.ylabel('Density')
plt.xlabel('Error')
plt.savefig(Error_Density_Histogramm_path_1)
plt.clf()

plt.hist(Seeds_Concatenated_Error_List_csv_1_non_nan, density=False, bins=80)  # density=False would make counts
plt.ylabel('Frequence')
plt.xlabel('Error')
plt.savefig(Seeds_Concatenated_Error_Histogramm_path_1)
plt.clf()

plt.hist(Seeds_Concatenated_Error_List_csv_1_non_nan, density=True, bins=80)  # density=False would make counts
plt.ylabel('Density')
plt.xlabel('Error')
plt.savefig(Seeds_Concatenated_Error_Density_Histogramm_path_1)
plt.clf()
#########################################################################################################################

Error_List_csv_2 = pd.read_csv("Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200818_xb_reaction2_6um009_testset/Error_List.csv",header=None)
Seeds_Concatenated_Error_List_csv_2 = pd.read_csv("Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200818_xb_reaction2_6um009_testset/Seeds_Concatenated_Error_List.csv",header=None)
Error_Histogramm_path_2 = "Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200818_xb_reaction2_6um009_testset/Error_Histogramm.png"
Error_Density_Histogramm_path_2 = "Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200818_xb_reaction2_6um009_testset/Error_Density_Histogramm.png"
Seeds_Concatenated_Error_Histogramm_path_2 = "Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200818_xb_reaction2_6um009_testset/Seeds_Concatenated_Error_Histogramm.png"
Seeds_Concatenated_Error_Density_Histogramm_path_2 = "Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200818_xb_reaction2_6um009_testset/Seeds_Concatenated_Error_Density_Histogramm.png"

Error_List_csv_2_non_nan = [x for x in Error_List_csv_2.values.flatten() if str(x) != 'nan']
Seeds_Concatenated_Error_List_csv_2_non_nan = [x for x in Seeds_Concatenated_Error_List_csv_2.values.flatten() if str(x) != 'nan']

Error_List_csv_2_larger_error = [x for x in Error_List_csv_2.values.flatten() if x > 10 ]
Seeds_Concatenated_Error_List_csv_2_larger_error = [x for x in Seeds_Concatenated_Error_List_csv_2.values.flatten() if x > 10]

error_average_2 = np.mean(Error_List_csv_2_non_nan)
error_min_2 = min(Error_List_csv_2_non_nan)
error_max_2 = max(Error_List_csv_2_non_nan)
error_var_2 = np.var(Error_List_csv_2_non_nan)

seed_concatenated_error_average_2 = np.mean(Seeds_Concatenated_Error_List_csv_2_non_nan)
seed_concatenated_error_min_2 = min(Seeds_Concatenated_Error_List_csv_2_non_nan)
seed_concatenated_error_max_2 = max(Seeds_Concatenated_Error_List_csv_2_non_nan)
seed_concatenated_error_var_2 = np.var(Seeds_Concatenated_Error_List_csv_2_non_nan)

print("2.Total data quantity",len(Error_List_csv_2_non_nan))
print("2.The error average: %f     2.The minimal error: %f     2.The maximal error: %f     2.The error variance: %f"%(error_average_2, error_min_2, error_max_2, error_var_2))
print("2.The error larger than 10 pixels quantity: ",len(Error_List_csv_2_larger_error))

print("2.Total seed concatenated error data quantity",len(Seeds_Concatenated_Error_List_csv_2.values.flatten()))
print("2.The seed concatenated error average: %f     2.The seed concatenated error minimal error: %f     2.The seed concatenated error maximal error: %f     2.The seed concatenated error variance: %f"%(seed_concatenated_error_average_2, seed_concatenated_error_min_2, seed_concatenated_error_max_2, seed_concatenated_error_var_2))
print("2.The seed concatenated error larger than 10 pixels quantity: " ,len(Seeds_Concatenated_Error_List_csv_2_larger_error))
print('\n')

plt.hist(Error_List_csv_2_non_nan, density=False, bins=80)  # density=False would make counts
plt.ylabel('Frequence')
plt.xlabel('Error')
plt.savefig(Error_Histogramm_path_2)
plt.clf()

plt.hist(Error_List_csv_2_non_nan, density=True, bins=80)  # density=False would make counts
plt.ylabel('Density')
plt.xlabel('Error')
plt.savefig(Error_Density_Histogramm_path_2)
plt.clf()

plt.hist(Seeds_Concatenated_Error_List_csv_2_non_nan, density=False, bins=80)  # density=False would make counts
plt.ylabel('Frequence')
plt.xlabel('Error')
plt.savefig(Seeds_Concatenated_Error_Histogramm_path_2)
plt.clf()

plt.hist(Seeds_Concatenated_Error_List_csv_2_non_nan, density=True, bins=80)  # density=False would make counts
plt.ylabel('Density')
plt.xlabel('Error')
plt.savefig(Seeds_Concatenated_Error_Density_Histogramm_path_2)
plt.clf()

#########################################################################################################################

Error_List_csv_3 = pd.read_csv("Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200820_xl_reaction6_03um001_testset/Error_List.csv",header=None)
Seeds_Concatenated_Error_List_csv_3 = pd.read_csv("Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200820_xl_reaction6_03um001_testset/Seeds_Concatenated_Error_List.csv",header=None)
Error_Histogramm_path_3 = "Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200820_xl_reaction6_03um001_testset/Error_Histogramm.png"
Error_Density_Histogramm_path_3 = "Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200820_xl_reaction6_03um001_testset/Error_Density_Histogramm.png"
Seeds_Concatenated_Error_Histogramm_path_3 = "Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200820_xl_reaction6_03um001_testset/Seeds_Concatenated_Error_Histogramm.png"
Seeds_Concatenated_Error_Density_Histogramm_path_3 = "Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200820_xl_reaction6_03um001_testset/Seeds_Concatenated_Error_Density_Histogramm.png"

Error_List_csv_3_non_nan = [x for x in Error_List_csv_3.values.flatten() if str(x) != 'nan']
Seeds_Concatenated_Error_List_csv_3_non_nan = [x for x in Seeds_Concatenated_Error_List_csv_3.values.flatten() if str(x) != 'nan']

Error_List_csv_3_larger_error = [x for x in Error_List_csv_3.values.flatten() if x > 10 ]
Seeds_Concatenated_Error_List_csv_3_larger_error = [x for x in Seeds_Concatenated_Error_List_csv_3.values.flatten() if x > 10]

error_average_3 = np.mean(Error_List_csv_3_non_nan)
error_min_3 = min(Error_List_csv_3_non_nan)
error_max_3 = max(Error_List_csv_3_non_nan)
error_var_3 = np.var(Error_List_csv_3_non_nan)

seed_concatenated_error_average_3 = np.mean(Seeds_Concatenated_Error_List_csv_3_non_nan)
seed_concatenated_error_min_3 = min(Seeds_Concatenated_Error_List_csv_3_non_nan)
seed_concatenated_error_max_3 = max(Seeds_Concatenated_Error_List_csv_3_non_nan)
seed_concatenated_error_var_3 = np.var(Seeds_Concatenated_Error_List_csv_3_non_nan)

print("3.Total data quantity",len(Error_List_csv_3_non_nan))
print("3.The error average: %f     3.The minimal error: %f     3.The maximal error: %f     3.The error variance: %f"%(error_average_3, error_min_3, error_max_3, error_var_3))
print("3.The error larger than 10 pixels quantity: ",len(Error_List_csv_3_larger_error))

print("3.Total seed concatenated error data quantity",len(Seeds_Concatenated_Error_List_csv_3.values.flatten()))
print("3.The seed concatenated error average: %f     3.The seed concatenated error minimal error: %f     3.The seed concatenated error maximal error: %f     3.The seed concatenated error variance: %f"%(seed_concatenated_error_average_3, seed_concatenated_error_min_3, seed_concatenated_error_max_3, seed_concatenated_error_var_3))
print("3.The seed concatenated error larger than 10 pixels quantity: " ,len(Seeds_Concatenated_Error_List_csv_3_larger_error))
print('\n')

plt.hist(Error_List_csv_3_non_nan, density=False, bins=80)  # density=False would make counts
plt.ylabel('Frequence')
plt.xlabel('Error')
plt.savefig(Error_Histogramm_path_3)
plt.clf()

plt.hist(Error_List_csv_3_non_nan, density=True, bins=80)  # density=False would make counts
plt.ylabel('Density')
plt.xlabel('Error')
plt.savefig(Error_Density_Histogramm_path_3)
plt.clf()

plt.hist(Seeds_Concatenated_Error_List_csv_3_non_nan, density=False, bins=80)  # density=False would make counts
plt.ylabel('Frequence')
plt.xlabel('Error')
plt.savefig(Seeds_Concatenated_Error_Histogramm_path_3)
plt.clf()

plt.hist(Seeds_Concatenated_Error_List_csv_3_non_nan, density=True, bins=80)  # density=False would make counts
plt.ylabel('Density')
plt.xlabel('Error')
plt.savefig(Seeds_Concatenated_Error_Density_Histogramm_path_3)
plt.clf()

#########################################################################################################################

Error_List_csv_4 = pd.read_csv("Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200824_xl_reaction2_03um006_testset/Error_List.csv",header=None)
Seeds_Concatenated_Error_List_csv_4 = pd.read_csv("Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200824_xl_reaction2_03um006_testset/Seeds_Concatenated_Error_List.csv",header=None)
Error_Histogramm_path_4 = "Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200824_xl_reaction2_03um006_testset/Error_Histogramm.png"
Error_Density_Histogramm_path_4 = "Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200824_xl_reaction2_03um006_testset/Error_Density_Histogramm.png"
Seeds_Concatenated_Error_Histogramm_path_4 = "Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200824_xl_reaction2_03um006_testset/Seeds_Concatenated_Error_Histogramm.png"
Seeds_Concatenated_Error_Density_Histogramm_path_4 = "Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200824_xl_reaction2_03um006_testset/Seeds_Concatenated_Error_Density_Histogramm.png"

Error_List_csv_4_non_nan = [x for x in Error_List_csv_4.values.flatten() if str(x) != 'nan']
Seeds_Concatenated_Error_List_csv_4_non_nan = [x for x in Seeds_Concatenated_Error_List_csv_4.values.flatten() if str(x) != 'nan']

Error_List_csv_4_larger_error = [x for x in Error_List_csv_4.values.flatten() if x > 10 ]
Seeds_Concatenated_Error_List_csv_4_larger_error = [x for x in Seeds_Concatenated_Error_List_csv_4.values.flatten() if x > 10]

error_average_4 = np.mean(Error_List_csv_4_non_nan)
error_min_4 = min(Error_List_csv_4_non_nan)
error_max_4 = max(Error_List_csv_4_non_nan)
error_var_4 = np.var(Error_List_csv_4_non_nan)

seed_concatenated_error_average_4 = np.mean(Seeds_Concatenated_Error_List_csv_4_non_nan)
seed_concatenated_error_min_4 = min(Seeds_Concatenated_Error_List_csv_4_non_nan)
seed_concatenated_error_max_4 = max(Seeds_Concatenated_Error_List_csv_4_non_nan)
seed_concatenated_error_var_4 = np.var(Seeds_Concatenated_Error_List_csv_4_non_nan)

print("4.Total data quantity",len(Error_List_csv_4_non_nan))
print("4.The error average: %f     4.The minimal error: %f     4.The maximal error: %f     4.The error variance: %f"%(error_average_4, error_min_4, error_max_4, error_var_4))
print("4.The error larger than 10 pixels quantity: ",len(Error_List_csv_4_larger_error))

print("4.Total seed concatenated error data quantity",len(Seeds_Concatenated_Error_List_csv_4.values.flatten()))
print("4.The seed concatenated error average: %f     4.The seed concatenated error minimal error: %f     4.The seed concatenated error maximal error: %f     4.The seed concatenated error variance: %f"%(seed_concatenated_error_average_4, seed_concatenated_error_min_4, seed_concatenated_error_max_4, seed_concatenated_error_var_4))
print("4.The seed concatenated error larger than 10 pixels quantity: " ,len(Seeds_Concatenated_Error_List_csv_4_larger_error))
print('\n')

plt.hist(Error_List_csv_4_non_nan, density=False, bins=80)  # density=False would make counts
plt.ylabel('Frequence')
plt.xlabel('Error')
plt.savefig(Error_Histogramm_path_4)
plt.clf()

plt.hist(Error_List_csv_4_non_nan, density=True, bins=80)  # density=False would make counts
plt.ylabel('Density')
plt.xlabel('Error')
plt.savefig(Error_Density_Histogramm_path_4)
plt.clf()

plt.hist(Seeds_Concatenated_Error_List_csv_4_non_nan, density=False, bins=80)  # density=False would make counts
plt.ylabel('Frequence')
plt.xlabel('Error')
plt.savefig(Seeds_Concatenated_Error_Histogramm_path_4)
plt.clf()

plt.hist(Seeds_Concatenated_Error_List_csv_4_non_nan, density=True, bins=80)  # density=False would make counts
plt.ylabel('Density')
plt.xlabel('Error')
plt.savefig(Seeds_Concatenated_Error_Density_Histogramm_path_4)
plt.clf()

#########################################################################################################################

Error_List_csv_5 = pd.read_csv("Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200825_xb_reaction3_03um007_testset/Error_List.csv",header=None)
Seeds_Concatenated_Error_List_csv_5 = pd.read_csv("Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200825_xb_reaction3_03um007_testset/Seeds_Concatenated_Error_List.csv",header=None)
Error_Histogramm_path_5 = "Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200825_xb_reaction3_03um007_testset/Error_Histogramm.png"
Error_Density_Histogramm_path_5 = "Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200825_xb_reaction3_03um007_testset/Error_Density_Histogramm.png"
Seeds_Concatenated_Error_Histogramm_path_5 = "Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200825_xb_reaction3_03um007_testset/Seeds_Concatenated_Error_Histogramm.png"
Seeds_Concatenated_Error_Density_Histogramm_path_5 = "Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/200825_xb_reaction3_03um007_testset/Seeds_Concatenated_Error_Density_Histogramm.png"

Error_List_csv_5_non_nan = [x for x in Error_List_csv_5.values.flatten() if str(x) != 'nan']
Seeds_Concatenated_Error_List_csv_5_non_nan = [x for x in Seeds_Concatenated_Error_List_csv_5.values.flatten() if str(x) != 'nan']

Error_List_csv_5_larger_error = [x for x in Error_List_csv_5.values.flatten() if x > 10 ]
Seeds_Concatenated_Error_List_csv_5_larger_error = [x for x in Seeds_Concatenated_Error_List_csv_5.values.flatten() if x > 10]

error_average_5 = np.mean(Error_List_csv_5_non_nan)
error_min_5 = min(Error_List_csv_5_non_nan)
error_max_5 = max(Error_List_csv_5_non_nan)
error_var_5 = np.var(Error_List_csv_5_non_nan)

seed_concatenated_error_average_5 = np.mean(Seeds_Concatenated_Error_List_csv_5_non_nan)
seed_concatenated_error_min_5 = min(Seeds_Concatenated_Error_List_csv_5_non_nan)
seed_concatenated_error_max_5 = max(Seeds_Concatenated_Error_List_csv_5_non_nan)
seed_concatenated_error_var_5 = np.var(Seeds_Concatenated_Error_List_csv_5_non_nan)

print("5.Total data quantity",len(Error_List_csv_5_non_nan))
print("5.The error average: %f     5.The minimal error: %f     5.The maximal error: %f     5.The error variance: %f"%(error_average_5, error_min_5, error_max_5, error_var_5))
print("5.The error larger than 10 pixels quantity: ",len(Error_List_csv_5_larger_error))

print("5.Total seed concatenated error data quantity",len(Seeds_Concatenated_Error_List_csv_5.values.flatten()))
print("5.The seed concatenated error average: %f     5.The seed concatenated error minimal error: %f     5.The seed concatenated error maximal error: %f     5.The seed concatenated error variance: %f"%(seed_concatenated_error_average_5, seed_concatenated_error_min_5, seed_concatenated_error_max_5, seed_concatenated_error_var_5))
print("5.The seed concatenated error larger than 10 pixels quantity: " ,len(Seeds_Concatenated_Error_List_csv_5_larger_error))
print('\n')

plt.hist(Error_List_csv_5_non_nan, density=False, bins=80)  # density=False would make counts
plt.ylabel('Frequence')
plt.xlabel('Error')
plt.savefig(Error_Histogramm_path_5)
plt.clf()

plt.hist(Error_List_csv_5_non_nan, density=True, bins=80)  # density=False would make counts
plt.ylabel('Density')
plt.xlabel('Error')
plt.savefig(Error_Density_Histogramm_path_5)
plt.clf()

plt.hist(Seeds_Concatenated_Error_List_csv_5_non_nan, density=False, bins=80)  # density=False would make counts
plt.ylabel('Frequence')
plt.xlabel('Error')
plt.savefig(Seeds_Concatenated_Error_Histogramm_path_5)
plt.clf()

plt.hist(Seeds_Concatenated_Error_List_csv_5_non_nan, density=True, bins=80)  # density=False would make counts
plt.ylabel('Density')
plt.xlabel('Error')
plt.savefig(Seeds_Concatenated_Error_Density_Histogramm_path_5)
plt.clf()


#########################################################################################################################

General_Error_data = []
for i in Error_List_csv_1_non_nan:
    General_Error_data.append(i)
for j in Error_List_csv_2_non_nan:
    General_Error_data.append(j)
for k in Error_List_csv_3_non_nan:
    General_Error_data.append(k)
for l in Error_List_csv_4_non_nan:
    General_Error_data.append(l)
for m in Error_List_csv_5_non_nan:
    General_Error_data.append(m)

General_Error_min = min(General_Error_data)
General_Error_max = max(General_Error_data)
General_Error_mean = np.mean(General_Error_data)
General_Error_var = np.var(General_Error_data)
General_Larger_10_Pixel =  sum(w > 10 for w in General_Error_data)

print("General minimal error: ",General_Error_min)
print("General maximal error: ",General_Error_max)
print("General error mean: ",General_Error_mean)
print("General error variance: ",General_Error_var)
print("General error larger than 10 pixels:",General_Larger_10_Pixel)


General_Seed_Concatenated_Error_data = []
for n in Seeds_Concatenated_Error_List_csv_1_non_nan:
    General_Seed_Concatenated_Error_data.append(n)
for o in Seeds_Concatenated_Error_List_csv_2_non_nan:
    General_Seed_Concatenated_Error_data.append(o)
for p in Seeds_Concatenated_Error_List_csv_3_non_nan:
    General_Seed_Concatenated_Error_data.append(p)
for q in Seeds_Concatenated_Error_List_csv_4_non_nan:
    General_Seed_Concatenated_Error_data.append(q)
for r in Seeds_Concatenated_Error_List_csv_5_non_nan:
    General_Seed_Concatenated_Error_data.append(r)

General_Seed_Concatenated_Error_min = min(General_Seed_Concatenated_Error_data)
General_Seed_Concatenated_Error_max = max(General_Seed_Concatenated_Error_data)
General_Seed_Concatenated_Error_mean = np.mean(General_Seed_Concatenated_Error_data)
General_Seed_Concatenated_Error_var = np.var(General_Seed_Concatenated_Error_data)
General_Seed_Concatenated_Larger_10_Pixel =  sum(v > 10 for v in General_Seed_Concatenated_Error_data)

print("General seed concatenated minimal error: ",General_Seed_Concatenated_Error_min)
print("General seed concatenated maximal error: ",General_Seed_Concatenated_Error_max)
print("General seed concatenated error mean: ",General_Seed_Concatenated_Error_mean)
print("General seed concatenated error variance: ",General_Seed_Concatenated_Error_var)
print("General seed concatenated error larger than 10 pixels:",General_Seed_Concatenated_Larger_10_Pixel)


plt.hist(General_Error_data, density=False, bins=200)  # density=False would make counts
plt.ylabel('Frequence')
plt.xlabel('Error')
plt.savefig("Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/General_Error_Histogramm")
plt.clf()

plt.hist(General_Error_data, density=True, bins=200)  # density=False would make counts
plt.ylabel('Density')
plt.xlabel('Error')
plt.savefig("Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/General_Error_Density_Histogramm")
plt.clf()

plt.hist(General_Seed_Concatenated_Error_data, density=False, bins=200)  # density=False would make counts
plt.ylabel('Frequence')
plt.xlabel('Error')
plt.savefig("Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/General_Seeds_Concatenated_Error_Histogramm")
plt.clf()

plt.hist(General_Seed_Concatenated_Error_data, density=True, bins=200)  # density=False would make counts
plt.ylabel('Density')
plt.xlabel('Error')
plt.savefig("Semantic_Segmentation/implementation/Testset_full_auto/Testset_Labels/General_Seeds_Concatenated_Error_Density_Histogramm")
plt.clf()