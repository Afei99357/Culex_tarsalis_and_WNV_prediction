import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import kendalltau
import seaborn as sns
import matplotlib.patches as mpatches

# Load the dataset into a Pandas DataFrame

data_origin = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/"
                          "CA_13_county_dataset/CA_13_counties_04_23_no_impute_daylight.csv",
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
    "Latitude",
    "Longitude",
    "Date",
    # "average_human_case_monthly",
    "Mos_WNV_Count",
    "Total_Bird_WNV_Count",
    "Horse_WNV_Count"
], axis=1)

## drop the columns if all the values in the columns are the same or all nan
data = data.dropna(axis=1, how='all')

## impute any missing in Human_Disease_Count with 0
data["Human_Disease_Count"] = data["Human_Disease_Count"].fillna(0)

## drop rows if has nan values
data = data.dropna()

## move the target column to the last column
data = data[[c for c in data if c not in ['Human_Disease_Count']] + ['Human_Disease_Count']]

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

## plot the heatmap again
## create a dict to store the column names, index using numbers, start from 0, key use the name of column name
column_names_dict_new = {}
for i in range(len(kendall_corr.columns)):
    column_names_dict_new[i] = kendall_corr.columns[i]

labels = list(column_names_dict_new.keys())

fig, ax = plt.subplots(figsize=(55, 45))

df = pd.DataFrame(kendall_pvalues.to_numpy(), index=labels, columns=labels)

## Create heatmap of Correlation
sns.heatmap(df, cmap='PiYG', annot=True, annot_kws={"size": 18}, fmt=".2f", linewidths=0.5, ax=ax, center=0)

## get the legend using the name of column names
handles = [mpatches.Patch(color='none', label=f'{i}: {name}') for i, name in column_names_dict_new.items()]

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

plt.title("Kendall's Tau Test P-value Heatmap", fontdict={'fontsize': 30})

plt.subplots_adjust(hspace=0.05, top=0.95, left=0.03, bottom=0.05, right=0.77)

plt.savefig("/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/result/plots/heatmap_kendall_pvalue_cali_week_04_to_24_13_counties_impute_0_with_daylight.png", dpi=300)

plt.show()



