##### run PCA on the data #####
#
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/"
                   "human/cdc_human_1999_to_2023/WNV_human_and_non_human_yearly_climate_demographic_bird.csv", index_col=0)

## remove any space and comma in Population column
data["Population"] = data["Population"].str.replace(",", "").str.strip()

## convert Population column to numeric
data["Population"] = pd.to_numeric(data["Population"], errors='coerce')

# select the columns after column Date as predictors
date_index = data.columns.get_loc("Date")

# Select columns after the "Date" column as predictors
data_pred = data.iloc[:, date_index+1:]

# get Year column from data and add it to data_pred
data_pred['Year'] = data['Year']

## add target column
data_pred["Neuroinvasive_disease_cases"] = data["Neuroinvasive_disease_cases"]

# drop nan values
data_pred = data_pred.dropna()

## reset index
data_pred = data_pred.reset_index(drop=True)

## preprocess the data
scaler = StandardScaler()

#create copy of DataFrame
scaled_data = data_pred.copy()

#created scaled version of DataFrame
scaled_df = pd.DataFrame(scaler.fit_transform(scaled_data), columns=scaled_data.columns)

## drop the target column
scaled_df = scaled_df.drop("Neuroinvasive_disease_cases", axis=1)

column_list = scaled_df.columns.values.tolist()

## run PCA on the data
pca = PCA()

## fit the data
pca.fit(scaled_df)

## plot scree plot
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')

loadings_matrix = pca.components_
print(pca.explained_variance_ratio_)
plt.savefig("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/result/PCA/human_yearly_cdc_pca_scree_plot.png", dpi=300)
plt.show()


# store the loadings matrix and top 6 variables for each component, then find the common variables in all components
pc_names = {}

for i in range(10):
    component_i_loadings = loadings_matrix[i]
    variable_indices = np.argsort(np.abs(component_i_loadings))[::-1]
    variable_names = [column_list[j] for j in variable_indices]
    pc_names[i] = variable_names[:6]
    print(f"Principal component {i+1}: {variable_names[:6]}")

## correlation bwtween each principal component and the target
# get the principal components
pca_data = pca.transform(scaled_df)

# get the target
target = data_pred["Neuroinvasive_disease_cases"]

# get the correlation between each principal component and the target
correlation = []
for i in range(pca_data.shape[1]):
    corr = np.corrcoef(pca_data[:, i], target)[0, 1]
    correlation.append(corr)

# plot the correlation
plt.bar(np.arange(len(correlation)), correlation)
plt.title("Correlation between principal components and target")
plt.xlabel("Principal Component")
plt.ylabel("Correlation")
plt.savefig("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/result/PCA/human_yearly_cdc_pca_corr_pc_and_target.png", dpi=300)
plt.show()



