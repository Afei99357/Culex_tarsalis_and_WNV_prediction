import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# Load data
df = pd.read_csv("/Users/ericliao/Desktop/poisson_mixed_model_states_model_coef_table.csv", index_col=False)

# Get the first column values where the column name is empty
features = list(df.iloc[:, 0])

# Drop the first column
df = df.drop(df.columns[0], axis=1)

# Get the column names
states = list(df.columns)

fig, ax = plt.subplots(figsize=(30, 25))

# Transpose the DataFrame
df_transposed = df.T

# Create heatmap of Correlation
sns.heatmap(df_transposed, cmap='Reds', annot=True, annot_kws={"size": 18}, fmt=".2f", linewidths=0.5, ax=ax)

# Set custom tick labels for the x-axis (previously y-axis)
plt.xticks(np.arange(len(features)) + 0.5, features, rotation='vertical')

# Set custom tick labels for the y-axis (previously x-axis)
plt.yticks(np.arange(len(states)) + 0.5, states, rotation=0, fontsize=25)

# X-axis tick font
plt.xticks(fontsize=25)

# Y-axis tick font
plt.yticks(fontsize=25)

# Color bar font size
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=25)

plt.title("Coefficient of Each Fixed Effect Feature in Each Individual State Poisson Mixed Model", fontdict={'fontsize': 30})

plt.tight_layout()

# save the figure
plt.savefig("/Users/ericliao/Desktop/heatmap_coef_states_poisson_mixed_model.png", dpi=300)
plt.show()
