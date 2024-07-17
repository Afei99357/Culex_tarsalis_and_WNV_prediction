###### Run PCA test on the bird dataset and plot scree plot\

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from venn import venn

# read data
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/birds_data/birds_counts_all_states_for_PCA.csv", index_col=False)

data = data.dropna()

## make copy of data
data_copy = data.copy()

## pivot the data
df_pivot = pd.pivot_table(data_copy, values='SpeciesTotal', index=['StateNum', 'Route', 'Year'], columns=['English_Common_Name'],
                          aggfunc=np.sum, fill_value=0)

df_pivot = df_pivot.reset_index()

column_list = df_pivot.columns.values.tolist()

# # prepare for PCA, mean center and normalize the data
#define scaler
scaler = StandardScaler()

#create copy of DataFrame
scaled_data = df_pivot.copy()

scaled_data.drop(['StateNum', 'Route', 'Year'], axis=1, inplace=True)

#created scaled version of DataFrame
scaled_df = pd.DataFrame(scaler.fit_transform(scaled_data), columns=scaled_data.columns)

# # run PCA
pca = PCA(n_components=2)
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

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
#
# for i in range(6):
#     component_i_loadings = loadings_matrix[i]
#     variable_indices = np.argsort(np.abs(component_i_loadings))[::-1]
#     variable_names = [column_list[j] for j in variable_indices]
#     pc_names[i] = variable_names[:10]
#     print(f"Principal component {i+1}: {variable_names[:10]}")

# # plot venn diagram
# labels = ['PC1', 'PC2', 'PC3', 'PC4']
#
# sets = {labels[0]: set(pc_names[0]),
#         labels[1]: set(pc_names[1]),
#         labels[2]: set(pc_names[2]),
#         labels[3]: set(pc_names[3])}
#
# fig1, ax = plt.subplots(1, figsize=(10, 10))
# a = venn(sets, ax=ax)
# plt.legend(labels[:], ncol=6)
#
# plt.show()
#
# # find the common variables in all components
# common_variables = set(pc_names[0]).intersection(set(pc_names[1]), set(pc_names[2]), set(pc_names[3]))
# print(common_variables)

## find the unique variables between all components
#
# plot 3d
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(pca_data[:, 0], pca_data[:, 1], c='blue', alpha=0.5)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
# set log
ax.set_xscale('symlog')
ax.set_yscale('symlog')
# ax.set_zscale('symlog')
ax.set_title('PCA')
# interactive plot
plt.ion()
plt.show()

## print the scores for each bird species
scores = pd.DataFrame(pca.components_, columns=scaled_data.columns)
print(scores.iloc[0:10, 0:10])