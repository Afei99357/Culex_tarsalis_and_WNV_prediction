import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# read data
# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/Data/"
                   "California_wnv_count_weekly_2011_2012_peciticide_all_shift.csv", index_col=0)

# drop columns that are not features and drop target
data = data.drop(["State", "County", "Year", 'Month', 'Date', "County_Seat_Latitude", "County_Seat_Longitude", "FIPS", 'Population', 'Human_Disease_Count'], axis=1)

data = data.dropna()

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
