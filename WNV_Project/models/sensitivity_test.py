### use np.linalg.lstsq (least square linear regression) to do sensitivity analysis,
#### also plot the sensitivity analysis result #######

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read in the data
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_13_to_2023_02_20/"
                   "dataset/human_neuroinvasive_with_extreme_weather_with_county_seat_modify.csv", index_col=False)

# drop rows contains nan
data = data.dropna()

# find Counties in southern california
southern_california_counties = ["Los Angeles", "San Diego", "Orange", "Riverside", "San Bernardino", "Kern", "Ventura",
                                "Santa Barbara", "San Luis Obispo", "Imperial"]

data = data[data["County"].isin(southern_california_counties) | data["State"].isin(['North Dakota', 'South Dakota', 'Colorado'])]

# only on horse
# data = data[data["SET"] == "VET"]
# only on human
data = data[data["SET"] == "HUMAN"]

# split into train and test
year = data["Year"]

data = data.drop(["FIPS", "County", "State", "State_Code", "Year",
                  "County_Centroid_Latitude", "County_Centroid_Longitude", "County_Seat_Latitude",
                  "County_Seat_Longitude", "County_Seat", "Processed_Flag_Land_Use", 'SET',
                  'Poverty_Estimate_All_Ages'], axis=1)

## get the column u10_Jan and column swvl1_Dec index
column_u10_Jan_index = data.columns.get_loc("u10_Jan")
column_swvl1_Dec_index = data.columns.get_loc("swvl1_Dec")

## DROP the columns between column_u10_Jan and column_swvl1_Dec includes column_u10_Jan and column_swvl1_Dec
data = data.drop(data.columns[column_u10_Jan_index:column_swvl1_Dec_index + 1], axis=1)

## normalize each feature in the data
for column in data.columns:
    data[column] = (data[column] - data[column].mean()) / data[column].std()

wnv_count = data.pop("WNV_Count")

## create matrix m to store the sensitivity analysis Least-squares solution
# m = np.zeros((1, len(data.columns)))

## perform sensitivity analysis using np.linalg.lstsq on the data
m = np.linalg.lstsq(data, wnv_count, rcond=None)[0]

column_names_dict = {}
for i in range(len(data.columns)):
    column_names_dict[i] = data.columns[i]

labels = list(column_names_dict.keys())

## plot the sensitivity analysis result
plt.figure(figsize=(57, 30))
# plot the horizontal barchar of the sensitivity analysis result
plt.barh(data.columns, m)
# replace x axis labels with labels

# y axis ticks fontsize
plt.yticks(fontsize=25)
# x axis ticks fontsize
plt.xticks(fontsize=25)

plt.xscale("symlog")
plt.title("Sensitivity Analysis - Least-squares Solution Plot", fontsize=55)
plt.savefig("/Users/ericliao/Desktop/sensitivity_analysis.png", dpi=300)






