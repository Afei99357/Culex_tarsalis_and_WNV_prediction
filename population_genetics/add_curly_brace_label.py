# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

## READ TXT FILE WITH space and there is no column names
df_group = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/admixture/"
    "plink_file_for_admixture/culex_plink_new.4.Q",
    sep=" ",
    header=None,
)

## read file with mosquito ID and population
df_pop = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/"
    "Ctarsalis_sample_w_GPS_climate_average_new_filtered_id_region.csv",
    sep=",",
    header=0,
    index_col=0,
)

## get the State, City, region, vcfID, GPS.Lat, GPS.Lon from the df_pop and add it to the df_group
state_list = df_pop["State"].tolist()
city_list = df_pop["City"].tolist()
region_list = df_pop["region"].tolist()
vcfID_list = df_pop["vcfID"].tolist()
GPS_Lat_list = df_pop["GPS.Lat"].tolist()
GPS_Lon_list = df_pop["GPS.Lon"].tolist()
df_group["State"] = state_list
df_group["City"] = city_list
df_group["region"] = region_list
df_group["vcfID"] = vcfID_list
df_group["GPS.Lat"] = GPS_Lat_list
df_group["GPS.Lon"] = GPS_Lon_list

## add column names to the df_group which are group1, group2, group3, group4
df_group.columns = [
    "Northwest",
    "Midwest",
    "West Coast",
    "Southwest",
    "State",
    "City",
    "region",
    "vcfID",
    "GPS.Lat",
    "GPS.Lon",
]

## plot one stack bar plot with axis is each vcfID and the bar is the values of group1, group2, group3, group4, sorting the
## plot by GPS.lon first and then GPS.Lat, color the bar by the region, where northwest is skyblue,
## midwest is hotpink, west coast is goldenrod, southwest is forestgreen

# Sort the df_group by region and GPS.Lon in descending way
df_group = df_group.sort_values(by=["region", "GPS.Lon"], ascending=False)

# Create a new figure and axis for the plot
fig, ax = plt.subplots(figsize=(20, 7))

# The x locations for the groups - centered on each tick mark
ind = np.arange(len(df_group))

# Adjust the width of the bars to fit the figure width more closely
# This value might need fine-tuning depending on the number of bars and the desired appearance
bar_width = 0.8  # Adjust this to change how much of the x-axis is covered by bars

# Bottom position for each bar start
bottom = np.zeros(len(df_group))

# Colors for each stack
colors = ["skyblue", "hotpink", "goldenrod", "forestgreen"]
regions = ["Northwest", "Midwest", "West Coast", "Southwest"]

for color, region in zip(colors, regions):
    ax.bar(ind, df_group[region], bar_width, bottom=bottom, color=color, edgecolor='none')
    bottom += df_group[region].values

# Add curly braces for grouping cities by regions
for region in df_group["region"].unique():
    region_data = df_group[df_group["region"] == region]
    start_index = region_data.index[0]
    end_index = region_data.index[-1]
    ax.annotate("", xy=(start_index, -0.2), xytext=(end_index + 1, -0.2),
                arrowprops=dict(arrowstyle="-", lw=2))
    ax.text((start_index + end_index + 1) * 0.5, -0.5, region, ha='center', va='center', fontsize=12, rotation=90)

# Adjust the x-axis limits to better utilize the space
ax.set_xlim(-0.5, len(df_group) - 0.5)  # Adjusting limits to fit the bar widths

# Remove x and y tick labels
ax.set_xticks([])
ax.set_yticks([])

# Remove the outer frame
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

fig.subplots_adjust(left=0.05, right=0.95)  # Adjust as needed to fit the figure

# plt.savefig("/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/Admixture/stack_bar_plot_admixture_K_4.png", dpi=300)

plt.show()
