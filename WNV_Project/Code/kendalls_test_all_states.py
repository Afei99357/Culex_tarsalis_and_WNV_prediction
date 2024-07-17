import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import kendalltau
import seaborn as sns

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/"
                   "human_neuroinvasive_wnv_ebirds.csv", index_col=0)


### southern_california_counties = ["Los Angeles", "San Diego", "Orange", "Riverside", "San Bernardino", "Kern", "Ventura","Santa Barbara", "San Luis Obispo", "Imperial"]

### in FIPS
southern_california_counties = [6037, 6073, 6059, 6065, 6071, 6029, 6111, 6083, 6079, 6025]

data = data[data["FIPS"].isin(southern_california_counties) | data["State"].isin(['North Dakota', 'South Dakota', 'Colorado'])]

# data = data[data["FIPS"].isin(southern_california_counties)]

# # drop columns that are not features and drop target
data = data.drop(["FIPS", "County", "State", "State_Code", "Year", 'SET', "County_Seat", "County_Seat_Latitude",
                  "County_Seat_Longitude", "County_Centroid_Latitude", "County_Centroid_Longitude",
                  'Poverty_Estimate_All_Ages', "Population", "State_Land_Area", "Land_Change_Count_Since_1992",
                  "Land_Use_Class", "Processed_Flag_Land_Use", "WNV_Rate_Neural_With_All_Years",
                  # "WNV_Rate_Neural_Without_99_21",
                  # "WNV_Rate_Non_Neural_Without_99_21",
                  "State_Horse_WNV_Rate", "WNV_Rate_Non_Neural_Without_99_21_log",
                  "WNV_Rate_Neural_Without_99_21_log"# target column
                  ], axis=1)


### drop monthly weather data block #######################
## get the column u10_Jan and column swvl1_Dec index
column_Poverty_index = data.columns.get_loc("Poverty_Rate_Estimate_All_Ages")
column_u10_Jan_index = data.columns.get_loc("u10_Jan")
column_swvl1_Dec_index = data.columns.get_loc("swvl1_Dec")
column_tp_acc_extrem_index = data.columns.get_loc("tp_acc_Oct_to_Aug")

## DROP the columns between column_u10_Jan and column_swvl1_Dec includes column_u10_Jan and column_swvl1_Dec
# data = data.drop(data.columns[column_u10_Jan_index:column_swvl1_Dec_index + 1], axis=1)
data = data.drop(data.columns[column_u10_Jan_index:column_swvl1_Dec_index + 1], axis=1)
################################################################

data = data.dropna()

col_names = list(data.columns)
#
# # Calculate Kendall's rank correlation matrix and p-values
# kendall_pvalues = pd.DataFrame(np.zeros((data.shape[1], data.shape[1])), columns=data.columns, index=data.columns)
# kendall_corr = pd.DataFrame(np.zeros((data.shape[1], data.shape[1])), columns=data.columns, index=data.columns)
# for i in range(len(col_names)):
#     for j in range(i+1, len(col_names)):
#         feature_1 = data[col_names[i]]
#         feature_2 = data[col_names[j]]
#         kendall_corr.iloc[i, j], kendall_pvalues.iloc[i, j] = kendalltau(feature_1, feature_2)
#         # scatter plot
#         plt.scatter(feature_1.values, feature_2.values)
#         plt.title(f"{col_names[i]} vs {col_names[j]}\n Kendall's Tau Test Correlation: {kendall_corr.iloc[i, j]:.3f}"
#                   f"\n Kendall's Pvalue: {kendall_pvalues.iloc[i, j]:.3f}")
#         plt.xlabel(col_names[i])
#         plt.ylabel(col_names[j])
#
#         # log scale
#         plt.xscale("log")
#         plt.yscale("log")
#
#         # add title name
#         plt.savefig(f"/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/"
#                     f"2023_02_13_to_2023_02_20/plots/correlation_coefficient/Kendall_test_feature_pair/"
#                     f"Pvalue_{kendall_pvalues.iloc[i, j]}_{col_names[i]}_vs_{col_names[j]}.png", dpi=300)
#
#         plt.close()



# Calculate Kendall's rank correlation matrix and p-values
kendall_pvalues = pd.DataFrame(np.zeros((data.shape[1], data.shape[1])), columns=data.columns, index=data.columns)
kendall_corr = pd.DataFrame(np.zeros((data.shape[1], data.shape[1])), columns=data.columns, index=data.columns)
for col1 in data.columns:
    for col2 in data.columns:
        kendall_corr.loc[col1, col2], kendall_pvalues.loc[col1, col2] = kendalltau(data[col1], data[col2])

###  create a dict to store the column names, index using numbers, start from 0, key use the name of column name
column_names_dict = {}
for i in range(len(data.columns)):
    column_names_dict[i] = data.columns[i]

labels = list(column_names_dict.keys())

fig, ax = plt.subplots(figsize=(45, 20))

df = pd.DataFrame(kendall_pvalues.to_numpy(), index=labels, columns=labels)

# Create heatmap of Correlation
sns.heatmap(df, cmap='Reds', annot=True, annot_kws={"size": 18}, fmt=".2f", linewidths=0.5, ax=ax)

# plt.text(19, -0.5, "Legend:", fontsize=25, rotation=0, c="black")
# ## add text as legend, for each row is a key and value in the dictionary
# for i in range(len(labels)):
#     plt.text(19, i*5/10, list(column_names_dict.items())[i], fontsize=25, rotation=0, c="black")

plt.text(26, -1, "Legend:", fontsize=25, rotation=0, c="black")
## add text as legend, for each row is a key and value in the dictionary
for i in range(len(labels)):
    plt.text(26, i, list(column_names_dict.items())[i], fontsize=25, rotation=0, c="black")

# x axis tick font
plt.xticks(fontsize=25)

# y axis tick font
plt.yticks(fontsize=25)

# # make the color bar font size
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=25)

plt.title("Kendall's Tau Test P-values Heatmap", fontdict={'fontsize': 30})

plt.savefig("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_13_to_2023_02_20/"
            "plots/correlation_coefficient/Kendall_test_feature_pair/heatmap_pvalues_with_ebirds_s_cali_s_n_dakota_colorado.png", dpi=300)

plt.show()