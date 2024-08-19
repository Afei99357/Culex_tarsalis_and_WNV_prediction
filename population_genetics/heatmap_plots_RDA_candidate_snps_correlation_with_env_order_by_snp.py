import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
file_path = "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/Landscape_genetics_GEA/RDA_Redundancy_Analysis/rda_analysis_candidates_correlation_to_env.csv"  # Update the path to your CSV file
data = pd.read_csv(file_path)

# ## separate the contig and snp in "snp" column by _
# data['contig'] = data['snp'].str.split('_').str[0]
# data['snp_loc'] = data['snp'].str.split('_').str[1]


## order the dataset by contig then snp
data = data.sort_values(by=['temperature'], ascending=True)

# Drop the 'axis' column
data.drop(columns=['axis', 'loading', 'predictor', 'correlation'], inplace=True)

## set the 'snp' column as the index
data.set_index('snp', inplace=True)

## transpose the data
data = data.T

# Create the heatmap
plt.figure(figsize=(50, 20))
sns.heatmap(data, cmap='coolwarm', annot=False)


## y axis is the snp, x axis is the variables, adding x and y axis tick labels
plt.xticks(rotation=90)
plt.yticks(rotation=0)

## font size of ticks label
plt.xticks(fontsize=10)
plt.yticks(fontsize=20)

plt.tight_layout()

plt.title('Heatmap of SNPs and Variables')
plt.savefig('/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/Landscape_genetics_GEA/RDA_Redundancy_Analysis/rda_analysis_candidates_correlation_to_env_heatmap_order_by_tem.png', dpi=300)
