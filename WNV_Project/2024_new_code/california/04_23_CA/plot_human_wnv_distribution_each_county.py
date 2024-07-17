import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from sklearn import ensemble, metrics
import pandas as pd
import numpy as np



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
    "average_human_case_monthly",
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
## add the year column to the data
data["Year"] = data["Date"].dt.year

## create a color map for the counties
colors = plt.cm.rainbow(np.linspace(0, 1, len(counties)))

## for each county, get the distribution of the WNV counts over the year from 2004 to 2023
for i, county in zip(range(len(counties)), counties):
    ## get the data for the county
    county_data = data[data["County"] == county]
    ## get the WNV counts for each year
    wnv_counts = county_data.groupby(county_data["Year"]).sum()
    ## plot the distribution
    plt.plot(wnv_counts.index, wnv_counts["Human_Disease_Count"], label=county, color=colors[i], linewidth=0.5)
    plt.xlabel("Year")
    plt.ylabel("Human Disease Count")

    ## keep x axis to be integer
    plt.xticks(np.arange(2004, 2024, 1), rotation=45, fontsize=5)

    plt.title("Human WNV Distribution, all counties in California, 2004 to 2023")


    ## save the last plot
    if county == counties[-1]:
        plt.legend(fontsize=3.6, loc='upper right')
        plt.savefig("/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/result/plots/county_level_disease_count_plot_04_to_24/all_county_wnv_counts_distribution.png", dpi=300)



