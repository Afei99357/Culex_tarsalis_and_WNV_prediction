import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/CA_human_data_2004_to_2023_final.csv",
                   index_col=False,
                   header=0)

## get the Date column to a new dataframe
date = data.pop("Date")
county = data.pop("County")


# drop columns that are not features and drop target
data = data.drop([
    "Year",
    'Month',
    'FIPS',
    "Latitude",
    "Longitude",
    "Total_Bird_WNV_Count",
    "Mos_WNV_Count",
    "Horse_WNV_Count",
], axis=1)

## drop the columns if all the values in the columns are the same or all nan
data = data.dropna(axis=1, how='all')

## reindex the data
data = data.reset_index(drop=True)

## print 0 variance columns
print(data.columns[data.var() == 0])

## check if any columns has zero variance and drop the columns
data = data.loc[:, data.var() != 0]

## add the Date and county column back to the data
data["Date"] = date
data["County"] = county

# convert "Date" column to datetime
data["Date"] = pd.to_datetime(data["Date"])

## drop rows if has nan values
data = data.dropna().reset_index(drop=True)

## for each county, get the distribution of the WNV counts over the year from 2004 to 2023
## get the unique counties
counties = data["County"].unique()
counties.sort()

## get the unique years and sort them
years = data["Date"].dt.year.unique()
years.sort()
## add the year and month column to the data
data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month

# Get the unique parameters
parameters = data.columns

## round up Human_Disease_Count to integer
data["Human_Disease_Count"] = data["Human_Disease_Count"].apply(np.ceil)

## remove "Date", "County", "Year" from the parameters
parameters = [parameter for parameter in parameters if parameter not in ["Date", "County", "Year", "Month"]]

## create a color map for the counties
colors = plt.cm.rainbow(np.linspace(0, 1, len(counties)))

months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

for month in range(1, 13):
    temp_data = data[data["Month"] == month]
    plt.figure(figsize=(10, 5))
    for i, county in zip(range(len(counties)), counties):
        temp_data_county = temp_data[(temp_data["County"] == county)]

        ## if the county has no data, skip the county
        if temp_data_county.shape[0] == 0:
            continue

        plt.plot(temp_data_county["Year"], temp_data_county["Human_Disease_Count"], color=colors[i], label=county)

        ## if this is the last county:
        if i == len(counties) - 1:
            plt.title(f"Human_Disease_Count, {county} county, {months[month - 1]}, 2004-2023 Sum")
            plt.xlabel("Year")
            plt.ylabel("Sum of Human Disease Count")
            plt.legend(fontsize=3.5, loc='upper right')
            ## keep x axis to be integer
            plt.xticks(np.arange(2004, 2024, 1), rotation=45, fontsize=5)
            plt.savefig(f"/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/result/plots/variable_change_over_month_each_year_each_county/all_county_huamn_wnv_counts_month_{month}.png", dpi=300)
            plt.close()

