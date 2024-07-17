import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import kendalltau
import seaborn as sns
# Load data
data = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/monthly/cdc_sum_organism_all_with_phylodiversity.csv",
    index_col=0)

# # # # choose state only in one state

# state_name ='california'
#
# data = data[data["State"] == state_name]

# # # choose states are in the list
# state_list = ["california",
#               # "colorado",
#               "idaho", "illinois", "louisiana", "new york", "ohio", "texas"]
# data = data[data["State"].isin(state_list)]

# drop columns that are not features and drop target
data = data.drop(["Year", 'Month', "FIPS", "County", "State",
                  'County_Seat',
                  "Latitude", "Longitude", 'Date',
                  'Binary_Target',
                  ], axis=1)


# drop columns where name contains "_2m_
# data = data.drop([col for col in data.columns if "_2m_" in col], axis=1)

col_names = list(data.columns)

# drop nan rows
data = data.dropna()

# # Calculate Kendall's rank correlation matrix and p-values
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

fig, ax = plt.subplots(figsize=(60, 30))

df = pd.DataFrame(kendall_corr.to_numpy(), index=labels, columns=labels)

# Create heatmap of Correlation
sns.heatmap(df, cmap='Reds', annot=True, annot_kws={"size": 18}, fmt=".2f", linewidths=0.5, ax=ax)

plt.text(14, -1, "Legend:", fontsize=25, rotation=0, c="black")
## add text as legend, for each row is a key and value in the dictionary
for i in range(len(labels)):
    plt.text(14, i, list(column_names_dict.items())[i], fontsize=25, rotation=0, c="black")

# x axis tick font
plt.xticks(fontsize=25)

# y axis tick font
plt.yticks(fontsize=25)

# add second y axis on right the same as the first one
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(ax.get_yticks())
ax2.set_yticklabels(ax.get_yticklabels(), size=25, rotation=-90)

# # make the color bar font size
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=25)

plt.title("Kendall's Tau Test Correlation Heatmap", fontdict={'fontsize': 30})
#
plt.savefig(f"/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/monthly/heatmap_correlation_all_organism_nation.png", dpi=300)

plt.show()