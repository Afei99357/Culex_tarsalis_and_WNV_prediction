import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/CA_human_data_2004_to_2023_final.csv",
                   index_col=False,
                   header=0)

# Get the Date column to a new dataframe
date = data.pop("Date")

# Drop columns that are not features and drop target
data = data.drop([
    "Year",
    'Month',
    "County",
    'FIPS',
    "Latitude",
    "Longitude",
    "Total_Bird_WNV_Count",
    "Mos_WNV_Count",
    "Horse_WNV_Count",
    "average_human_case_monthly",
], axis=1)

# Drop columns if all the values in the columns are the same or all nan
data = data.dropna(axis=1, how='all')

# If the column names contains space or /, replace the space with underscore
data.columns = data.columns.str.replace(" ", "_")
data.columns = data.columns.str.replace("/", "_")



# Reindex the data
data = data.reset_index(drop=True)

# Print 0 variance columns
print(data.columns[data.var() == 0])

# Check if any columns has zero variance and drop the columns
data = data.loc[:, data.var() != 0]

# Add the Date column back to the data
data["Date"] = date

# Convert "Date" column to datetime
data["Date"] = pd.to_datetime(data["Date"])

# Get the unique years and sort them
years = data["Date"].dt.year.unique()
years.sort()

# Drop rows if has nan values
data = data.dropna().reset_index(drop=True)

# Get the unique parameters
parameters = data.columns
parameters = parameters[parameters != "Date"]

# ANOVA test for each parameter to see if the parameter changes over the years
# Create a dataframe to store the p-values
p_values = pd.DataFrame(index=parameters, columns=["p_value"])

for parameter in parameters:
    # Create a list to store each year's data
    year_data = []

    for year in years:
        data_year = data[data["Date"].dt.year == year]
        data_year = data_year.drop("Date", axis=1)
        data_year = data_year[parameter]
        year_data.append(data_year)

    # ANOVA test
    f_statistic, p_value = f_oneway(*year_data)

    p_values.loc[parameter, "p_value"] = p_value

## print the p-values
print(p_values)
