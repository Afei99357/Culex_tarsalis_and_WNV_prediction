import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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

#created scaled version of DataFrame
scaled_df = pd.DataFrame(scaler.fit_transform(scaled_data), columns=scaled_data.columns)

# # run PCA
pca = PCA()
pca.fit(scaled_data)

# # plot scree plot
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')

loadings_matrix = pca.components_
print(pca.explained_variance_ratio_)
# plt.savefig("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_13_to_2023_02_20/plots/PCA_Scree_Plot_.png", dpi=300)
plt.show()

# store the loadings matrix and top 10 variables for each component, then find the common variables in all components
pc_names = {}

for i in range(6):
    component_i_loadings = loadings_matrix[i]
    variable_indices = np.argsort(np.abs(component_i_loadings))[::-1]
    variable_names = [column_list[j] for j in variable_indices]
    pc_names[i] = variable_names[:10]
    print(f"Principal component {i+1}: {variable_names[:10]}")


