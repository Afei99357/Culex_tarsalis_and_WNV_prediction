import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import kendalltau
import seaborn as sns
import matplotlib.patches as mpatches

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/Europe_data/europe_data_with_coordinates_landuse_climate.csv", index_col=0)

# get the environment data
data_env = data.iloc[:, 11:]

## check for zero variance
print(data_env.var())

# remove column has zero variance and high correlation
data_env = data_env.drop(columns=["Evergreen Broadleaf Trees", "Wetland", "Snow/Ice", "avg_smlt"])

data_env = data_env.dropna()

col_names = list(data_env.columns)

# Calculate Kendall's rank correlation matrix and p-values
kendall_pvalues = pd.DataFrame(np.zeros((data_env.shape[1], data_env.shape[1])), columns=data_env.columns, index=data_env.columns)
kendall_corr = pd.DataFrame(np.zeros((data_env.shape[1], data_env.shape[1])), columns=data_env.columns, index=data_env.columns)
for col1 in data_env.columns:
    for col2 in data_env.columns:
        kendall_corr.loc[col1, col2], kendall_pvalues.loc[col1, col2] = kendalltau(data_env[col1], data_env[col2])

###  create a dict to store the column names, index using numbers, start from 0, key use the name of column name
column_names_dict = {}
for i in range(len(data_env.columns)):
    column_names_dict[i] = data_env.columns[i]

labels = list(column_names_dict.keys())

fig, ax = plt.subplots(figsize=(45, 35))

df = pd.DataFrame(kendall_corr.to_numpy(), index=labels, columns=labels)

## Create heatmap of Correlation
sns.heatmap(df, cmap='PiYG', annot=True, annot_kws={"size": 18}, fmt=".2f", linewidths=0.5, ax=ax, center=0)

## get the legend using the name of column names
handles = [mpatches.Patch(color='none',
                          label=f'{i}: {name}') for i, name in column_names_dict.items()]
plt.legend(handles=handles,
           title="Environmental Variables",
           title_fontsize=30,
           fontsize=30,
           loc="upper right",
           bbox_to_anchor=(1.6, 0.8))

# x axis tick font
plt.xticks(fontsize=25)

# y axis tick font
plt.yticks(fontsize=25)

# # make the color bar font size
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=25)

plt.title("Kendall's Tau Test Correlation Heatmap", fontdict={'fontsize': 30})

plt.subplots_adjust(hspace=0.05, top=0.95, left=0.03, bottom=0.05, right=0.77)

plt.savefig(f"/Users/ericliao/Desktop/WNV_project_files/WNV/Europe_data/heatmap_corr_kendall_eur.png", dpi=300)

plt.show()
