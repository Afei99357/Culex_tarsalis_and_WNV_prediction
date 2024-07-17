import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import kendalltau
import seaborn as sns
import matplotlib.patches as mpatches

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/cdc_human_1999_to_2023/"
            "WNV_human_and_non_human_yearly_climate_demographic_bird.csv",
                   index_col=False,
                   header=0)

# drop columns that are not features and drop target
data = data.drop([
    'FIPS',
    "Year",
    'State',
    "County",
    "Latitude",
    "Longitude",
    "Activity",
    "Identified_by_Blood_Donor_Screening",
    "Total_Bird_WNV_Count",
    "WNV_Corvid_Count",
    "WNV_NonCorvid_Count",
    "Mos_WNV_Count",
    "Horse_WNV_Count",
    "average_human_case_over_20_years",
    "Date"
], axis=1)

## drop na values and reset the index, print out how many rows are dropped
print("total rows before drop na: ", data.shape[0])
print("total rows after drop na: ", data.dropna().shape[0] - data.shape[0])
data = data.dropna()
data = data.reset_index(drop=True)

## move the target column to the last column
data = data[[c for c in data if c not in ['Neuroinvasive_disease_cases']] + ['Neuroinvasive_disease_cases']]
data = data[[c for c in data if c not in ['Reported_human_cases']] + ['Reported_human_cases']]

## check if any columns has zero variance and drop the columns, print the column names
print(data.var() == 0)
data = data.drop(data.var()[data.var() == 0].index, axis=1)

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

fig, ax = plt.subplots(figsize=(70, 40))

df = pd.DataFrame(kendall_pvalues.to_numpy(), index=labels, columns=labels)

## Create heatmap of Correlation
sns.heatmap(df, cmap='PiYG', annot=True, annot_kws={"size": 18}, fmt=".2f", linewidths=0.5, ax=ax, center=0)

## get the legend using the name of column names
handles = [mpatches.Patch(color='none',
                          label=f'{i}: {name}') for i, name in column_names_dict.items()]
plt.legend(handles=handles,
           title="Prediction Variables",
           title_fontsize=30,
           fontsize=30,
           frameon=False,
           loc="upper right",
           bbox_to_anchor=(1.3, 1))

# x axis tick font
plt.xticks(fontsize=25)

# y axis tick font
plt.yticks(fontsize=25)

# # make the color bar font size
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=25)

plt.title("Kendall's Tau Test Pvalue Heatmap", fontdict={'fontsize': 30})

plt.subplots_adjust(hspace=0.05, top=0.95, left=0.03, bottom=0.05, right=0.77)

## tight layout
plt.tight_layout()

plt.savefig("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/result/heatmap_pvalue_kendall_human_yealy_national.png", dpi=300)

plt.show()
