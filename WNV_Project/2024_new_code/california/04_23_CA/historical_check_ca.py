import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# read the csv file
data_california = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/"
                              "new_data_2004_2023/CA_human_data_2004_to_2023_new_extract.csv", sep=",")

## convert the "Report_Date" column to datetime 14-May-04
data_california["Report_Date"] = pd.to_datetime(data_california["Report_Date"], format="%d-%b-%y")

## separate the date column to year, month, and day columns
data_california["Year"] = data_california["Report_Date"].dt.year
data_california["Month"] = data_california["Report_Date"].dt.month
data_california["Day"] = data_california["Report_Date"].dt.day

## convert the "County" column to lower case
data_california["County"] = data_california["County"].str.lower()

## remove the leading and trailing white spaces
data_california["County"] = data_california["County"].str.strip()

## create a new df to only keep County, Caes, Year, Month, Day
data_california_new = data_california[["County", "Cases", "Year", "Month"]]

# group by Year, Month, and county, and sum the number of cases
data_california_new = data_california_new.groupby(["Year", "Month", "County"]).sum().reset_index()

## based on the Year, Month, and County, create a new empty dataframe contains all months and years from 2004 to 2023 and all counties
## get all unique years
years = data_california_new["Year"].unique()

## get all unique counties
counties = data_california_new["County"].unique()

## create a new empty dataframe
new_empty = pd.DataFrame(columns=["Year", "Month", "County"])

## loop through all years, months, and counties
for year in years:
    for month in range(1, 13):
        for county in counties:
            new_empty = new_empty.append({"Year": year, "Month": month, "County": county}, ignore_index=True)

## merge the new data with the original data
data_california_new = pd.merge(new_empty, data_california_new, how="left", on=["Year", "Month", "County"])

## get the ration of nan values in the Cases column
print(data_california_new["Cases"].isna().sum() / data_california_new.shape[0])


## reorder all the columns as Year, Month, County, Cases
data = data_california_new[["Year", "Month", "County",  "Latitude", "Longitude", "Cases", "Avian Phylodiversity"]]

## rename the "Cases" column to "Human_Disease_Count"
data = data.rename(columns={"Cases": "Human_Disease_Count"})

## based on the year to sum the number of human cases in each county, then plot the yearly human cases change in each county from 2004 to 2023
data_yearly = data.groupby(["Year", "County"]).sum().reset_index()

## for each county, plot the yearly human cases change from 2004 to 2023


## create a color map for the counties
colors = plt.cm.rainbow(np.linspace(0, 1, len(counties)))


## get all unique counties
counties = data_yearly["County"].unique()
#
## loop through all counties
plt.figure()
for i, county in zip(range(len(counties)), counties):
    data_county = data_yearly[data_yearly["County"] == county]
    plt.plot(data_county["Year"], data_county["Human_Disease_Count"], label=county, color=colors[i])
    if county == counties[-1]:
        plt.legend(fontsize=3.5, loc='upper right')
        plt.xlabel("Year")
        plt.ylabel("Human Disease Count")
        ## keep x axis to be integer
        plt.xticks(np.arange(2004, 2024, 1), rotation=45, fontsize=5)
        plt.title("Yearly Human Disease Count Change in California")
        plt.savefig("/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/result/plots/SUM/county_cases_change_over_years/all_counties_yearly_human_disease_count_change.png", dpi=300)
        plt.close()

## 13 couties with the most human cases
counties_13 = ["fresno", "kern", "los angeles", "merced", "orange", "placer", "riverside", "sacramento", "san bernardino",
             "san joaquin", "solano", "stanislaus", "tulare"]

plt.figure()
colors_13 = plt.cm.tab20(np.linspace(0, 1, len(counties_13)))
for i, county in zip(range(len(counties_13)), counties_13):
    data_county = data_yearly[data_yearly["County"] == county]
    plt.plot(data_county["Year"], data_county["Human_Disease_Count"], label=county, color=colors_13[i])
    if county == counties_13[-1]:
        plt.legend(fontsize=3.5, loc='upper right')
        plt.xlabel("Year")
        plt.ylabel("Human Disease Count")
        ## keep x axis to be integer
        plt.xticks(np.arange(2004, 2024, 1), rotation=45, fontsize=5)
        plt.title("Yearly Human Disease Count Change in 13 Highest \n Human Cases Counties in California")
        plt.savefig("/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/result/plots/SUM/county_cases_change_over_years/13_highest_counties_yearly_human_disease_count_change.png", dpi=300)
        plt.close()



## loop through all counties save individual plots
for i, county in zip(range(len(counties)), counties):
    plt.figure()
    data_county = data_yearly[data_yearly["County"] == county]
    plt.plot(data_county["Year"], data_county["Human_Disease_Count"], label=county, color=colors[i])
    plt.legend(fontsize=3.5, loc='upper right')
    plt.xlabel("Year")
    plt.ylabel("Human Disease Count")
    ## keep x axis to be integer
    plt.xticks(np.arange(2004, 2024, 1), rotation=45, fontsize=5)
    plt.title("Yearly Human Disease Count Change in California")
    plt.savefig(f"/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/result/plots/county_cases_change_over_years/{county}_yearly_human_disease_count_change.png", dpi=300)
    plt.close()

## group by year, then plot the yearly change of human cases in California
data_sum = data_yearly.groupby("Year").sum().reset_index()

## new plot
plt.figure()

plt.plot(data_sum["Year"], data_sum["Human_Disease_Count"])

plt.xlabel("Year")
plt.ylabel("Human Disease Count")
plt.title("Yearly Human Disease Count Change in California")
plt.xticks(np.arange(2004, 2024, 1), rotation=45)
plt.savefig("/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/result/plots/county_cases_change_over_years/SUM/California_yearly_human_disease_count_change.png", dpi=300)




