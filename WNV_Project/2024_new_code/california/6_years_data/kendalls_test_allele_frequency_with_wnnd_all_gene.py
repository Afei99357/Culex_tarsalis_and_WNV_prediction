import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import kendalltau
import seaborn as sns
import matplotlib.patches as mpatches

# Load the dataset into a Pandas DataFrame

data_origin = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/data/"
                   "cali_week_wnnd_with_allele_frequency_all_gene.csv",
                   index_col=False,
                   header=0)

## make a copy of the data
data = data_origin.copy()

# drop columns that are not features and drop target
data = data.drop([
    "County",
    "Year",
    'Month',
    'FIPS',
    'State',
    "Latitude",
    "Longitude",
    'Population',
    'Land_Area_2010',
    "Poverty_Estimate_All_Ages",
    "Date",
    "American Goldfinch",
    "House Finch",
    "House Sparrow",
    'American Crow',
    'American Robin',
    "Human_WNND_Rate",
    "WNV_Mos_Count",
    "WNV_Corvid_Count",
    "WNV_NonCorvid_Count",
    "Total_Bird_WNV_Count"
], axis=1)

data = data.dropna()

## drop the first 19 columns
data = data.drop(data.columns[0:20], axis=1)

## move the target column to the last column
data = data[[c for c in data if c not in ['Human_WNND_Count']] + ['Human_WNND_Count']]

## print 0 variance columns
print(data.columns[data.var() == 0])

## check if any columns has zero variance and drop the columns
data = data.loc[:, data.var() != 0]

## print the number of columns that are dropped
print("The number of columns that are dropped for 0 variances is: ", len(data_origin.columns) - len(data.columns))

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

fig, ax = plt.subplots(figsize=(55, 45))

df = pd.DataFrame(kendall_corr.to_numpy(), index=labels, columns=labels)

## Create heatmap of Correlation
sns.heatmap(df, cmap='PiYG', annot=True, annot_kws={"size": 18}, fmt=".2f", linewidths=0.5, ax=ax, center=0)

## get the legend using the name of column names
handles = [mpatches.Patch(color='none',
                          label=f'{i}: {name}') for i, name in column_names_dict.items()]
plt.legend(handles=handles,
           title="Prediction Variables",
           title_fontsize=30,
           fontsize=30,
           loc="upper right",
           bbox_to_anchor=(1.55, 0.9))

# x axis tick font
plt.xticks(fontsize=25)

# y axis tick font
plt.yticks(fontsize=25)

# # make the color bar font size
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=25)

plt.title("Kendall's Tau Test Correlation Heatmap", fontdict={'fontsize': 30})

plt.subplots_adjust(hspace=0.05, top=0.95, left=0.03, bottom=0.05, right=0.77)

# plt.savefig("/Users/ericliao/Desktop/WNV_project_files/WNV/california/plots/heatmap_corr_kendall_allele_frequency_all_gene.png", dpi=300)

plt.show()

## for each column in the data, check what are the other columns that has correlation with it greater than 0.7,
# drop the least number of columns to make sure no two columns has correlation greater than 0.7
# create a list to store the column names that has correlation greater than 0.7 with other columns
# Assuming 'kendall_corr' is your correlation DataFrame
threshold = 0.7

while True:
    # Find the pair with the highest correlation
    upper_tri = kendall_corr.where(np.triu(np.ones(kendall_corr.shape), k=1).astype(bool))
    max_corr = upper_tri.stack().abs().max()

    if max_corr <= threshold:
        # Exit loop if no correlation exceeds the threshold
        break

    # Get the pair with the highest correlation
    max_pair = upper_tri.stack().abs().idxmax()

    # get the correlation between both the variables in the pair and the target column Human_WNND_Count
    corr_with_target = kendall_corr.loc[max_pair[0], "Human_WNND_Count"]
    corr_with_target_2 = kendall_corr.loc[max_pair[1], "Human_WNND_Count"]

    # Decide which column of the pair to drop
    # For example, drop the column that corr_with_target is smaller
    if corr_with_target > corr_with_target_2:
        drop_col = max_pair[1]
    else:
        drop_col = max_pair[0]

    # Drop one of the variables in the pair
    kendall_corr = kendall_corr.drop(drop_col, axis=1)
    kendall_corr = kendall_corr.drop(drop_col, axis=0)


# print the columns that are dropped
print("The columns that are dropped are: ", list(set(data.columns) - set(kendall_corr.columns)))

## print the number of columns that are dropped
print("The number of columns that are dropped is: ", len(set(data.columns) - set(kendall_corr.columns)))

## plot the heatmap again
## create a dict to store the column names, index using numbers, start from 0, key use the name of column name
column_names_dict_new = {}
for i in range(len(kendall_corr.columns)):
    column_names_dict_new[i] = kendall_corr.columns[i]

labels = list(column_names_dict_new.keys())

fig, ax = plt.subplots(figsize=(55, 45))

df = pd.DataFrame(kendall_corr.to_numpy(), index=labels, columns=labels)

## Create heatmap of Correlation
sns.heatmap(df, cmap='PiYG', annot=True, annot_kws={"size": 18}, fmt=".2f", linewidths=0.5, ax=ax, center=0)

## get the legend using the name of column names
handles = [mpatches.Patch(color='none',
                          label=f'{i}: {name}') for i, name in column_names_dict_new.items()]

plt.legend(handles=handles,
              title="Prediction Variables",
              title_fontsize=30,
              fontsize=30,
              loc="upper right",
              bbox_to_anchor=(1.55, 0.9))

# x axis tick font
plt.xticks(fontsize=25)

# y axis tick font
plt.yticks(fontsize=25)

# # make the color bar font size
cbar = ax.collections[0].colorbar

cbar.ax.tick_params(labelsize=25)

plt.title("Kendall's Tau Test Correlation Heatmap", fontdict={'fontsize': 30})

plt.subplots_adjust(hspace=0.05, top=0.95, left=0.03, bottom=0.05, right=0.77)

plt.savefig("/Users/ericliao/Desktop/WNV_project_files/WNV/california/plots/heatmap_corr_kendall_allele_frequency_all_gene_after_drop.png", dpi=300)

plt.show()

## get the index of column Total_Bird_WNV_Count
index = data_origin.columns.get_loc("Total_Bird_WNV_Count")

## subset the data_origin which only contains the columns that are before and include Total_Bird_WNV_Count
data_new = data_origin.iloc[:, 0:index + 1]

## the columns after Total_Bird_WNV_Count, if the column names are in the kendall_corr.columns, then add to the data_new
for column in data_origin.columns[index + 1:]:
    if column in kendall_corr.columns:
        data_new[column] = data_origin[column]

# save the new data to a csv file
data_new.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/data/"
            "cali_week_wnnd_with_allele_frequency_all_gene_after_drop_high_correlated.csv", index=False)



