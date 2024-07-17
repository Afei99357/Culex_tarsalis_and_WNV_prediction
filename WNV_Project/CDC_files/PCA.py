### run PCA on the national dataset

# import packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns

# load data
df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/"
                 "cdc_sum_organism_all_with_phylodiversity.csv",
                 index_col=0)

# remove all the comma in the data
df["Population"] = df["Population"].str.replace(",", "")
# convert population to float
df["Population"] = df["Population"].astype(float)

# drop the rows where the value is nan and reset the index
df = df.dropna()

# drop the columns that are not needed
df = df.drop(columns=["Year", "Month", "FIPS", "County", "State", "County_Seat", "Binary_Target", "Date", "Total_Organism_WNV_Count"])

# data preprocessing the df before PCA
# drop the columns that are not needed
df_preprocessed = df.drop(columns=['Longitude', 'Latitude'])

# get the column list
column_list = df_preprocessed.columns.values.tolist()

# preprocessing the df
scaler = StandardScaler()
scaler.fit(df_preprocessed)
df_preprocessed = scaler.transform(df_preprocessed)

# PCA
pca = PCA()
pca.fit(df_preprocessed)
df_preprocessed = pca.transform(df_preprocessed)

# # plot scree plot
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')

loadings_matrix = pca.components_
print(pca.explained_variance_ratio_)
plt.show()

# store the loadings matrix and top 10 variables for each component, then find the common variables in all components
pc_names = {}

for i in range(6):
    component_i_loadings = loadings_matrix[i]
    variable_indices = np.argsort(np.abs(component_i_loadings))[::-1]
    variable_names = [column_list[j] for j in variable_indices]
    pc_names[i] = variable_names[:10]
    print(f"Principal component {i+1}: {variable_names[:10]}")


# plot pc=2 PCA score plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_preprocessed[:, 0], y=df_preprocessed[:, 1])
# label the variance being explained in PC1 and PC2 on the x and y label
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2f})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2f})")
plt.title("PC1 vs PC2")
plt.show()








