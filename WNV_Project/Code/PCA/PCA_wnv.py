###### Run PCA test on the dataset and plot scree plot\

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau
import seaborn as sns

# read data
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/"
                   "human_neuroinvasive_wnv_ebirds.csv", index_col=0)


### in FIPS
southern_california_counties = [6037, 6073, 6059, 6065, 6071, 6029, 6111, 6083, 6079, 6025]

# data = data[data["FIPS"].isin(southern_california_counties) | data["State"].isin(['North Dakota', 'South Dakota', 'Colorado'])]

data = data[data["FIPS"].isin(southern_california_counties)]

# # drop columns that are not features and drop target
data = data.drop(["FIPS", "County", "State", "State_Code", 'SET', "County_Seat", "County_Seat_Latitude",
                  "County_Seat_Longitude", "County_Centroid_Latitude", "County_Centroid_Longitude",
                  'Poverty_Estimate_All_Ages', "Population", "State_Land_Area",
                  "Processed_Flag_Land_Use", "WNV_Rate_Neural_With_All_Years",
                  # "WNV_Rate_Neural_Without_99_21",
                  # "WNV_Rate_Non_Neural_Without_99_21",
                  "State_Horse_WNV_Rate", "WNV_Rate_Non_Neural_Without_99_21_log",
                  "WNV_Rate_Neural_Without_99_21_log"# target column
                  ], axis=1)


### drop monthly weather data block #######################
## get the column u10_Jan and column swvl1_Dec index
column_u10_Jan_index = data.columns.get_loc("u10_Jan")
column_swvl1_Dec_index = data.columns.get_loc("swvl1_Dec")

## DROP the columns between column_u10_Jan and column_swvl1_Dec includes column_u10_Jan and column_swvl1_Dec
data = data.drop(data.columns[column_u10_Jan_index:column_swvl1_Dec_index + 1], axis=1)
################################################################

data = data.dropna()
data_wnv = data.pop("WNV_Rate_Neural_Without_99_21")
data_year = data.pop("Year")
data_Land_Change = data.pop("Land_Change_Count_Since_1992")
data_land_use = data.pop("Land_Use_Class")

column_list = data.columns.values.tolist()

# # prepare for PCA, mean center and normalize the data
#define scaler
scaler = StandardScaler()

#create copy of DataFrame
scaled_data = data.copy()

## number of features
feature_number = scaled_data.shape[1]

#created scaled version of DataFrame
scaled_df = pd.DataFrame(scaler.fit_transform(scaled_data), columns=scaled_data.columns)

# # run PCA
pca = PCA()
pca.fit(scaled_data)

# # plot scree plot
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title(f'Scree Plot (Total {feature_number} features)')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')

print(pca.explained_variance_ratio_)
plt.savefig("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_13_to_2023_02_20/plots/PCA_Scree_Plot_ebirds_sc_cali.png", dpi=300)
plt.show()

# # plot PCA
number_of_components = 2
pca = PCA(n_components=number_of_components)
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

kendall_pvalues = []
kendall_corr = []

for i in range(pca_data.shape[1]):
    corr, pvalue = kendalltau(pca_data[:, i], data_wnv.values)
    kendall_corr.append(corr)
    kendall_pvalues.append(pvalue)

# print the correlation and pvalue for each principal component with the target
for i in range(pca_data.shape[1]):
    print(f"PC{i+1}: corr: {kendall_corr[i]}, P-value: {kendall_pvalues[i]}")

loadings_matrix = pca.components_

for i in range(number_of_components):
    component_i_loadings = loadings_matrix[i]
    variable_indices = np.argsort(np.abs(component_i_loadings))[::-1]
    variable_names = [column_list[j] for j in variable_indices]
    print(f"Principal component {i+1}: {variable_names[:10]}")

# output the 4 PCs pca_data with target and year as csv
pca_data = pd.DataFrame(pca_data)
pca_data["WNV_Rate_Neural_Without_99_21"] = data_wnv.values
pca_data["Year"] = data_year.values
pca_data["Land_Change_Count_Since_1992"] = data_Land_Change.values
pca_data["Land_Use_Class"] = data_land_use.values
pca_data.to_csv(f"/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/PCA_{number_of_components}pc_wnv_s_cali_ebirds.csv")

# # plot
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(pca_data[:, 0], pca_data[:, 1], c='blue', alpha=0.5)
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_title('PCA')
# plt.show()