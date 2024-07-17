from sklearn import ensemble, metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/CA_human_data_2004_to_2023_final_impute_entire_04_23.csv",
                   index_col=False,
                   header=0)

##n choose data only in county: Fresno, Kern, Los Angeles, Merced, Orange, Placer, Riverside, Sacramento, San Bernardino, San Joaquin, Solano, Stanislaus, and Tulare
data = data[data["County"].isin(x.lower() for x in ["Fresno", "Kern", "Los Angeles", "Merced", "Orange", "Placer", "Riverside",
                                 "Sacramento", "San Bernardino", "San Joaquin", "Solano",
                                 "Stanislaus", "Tulare"])]

## reindex the data
data = data.reset_index(drop=True)

## get the Date column to a new dataframe
date = data.pop("Date")

# drop columns that are not features and drop target
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

## drop the columns if all the values in the columns are the same or all nan
data = data.dropna(axis=1, how='all')

## if the column names contains space or /, replace the space with underscore
data.columns = data.columns.str.replace(" ", "_")
data.columns = data.columns.str.replace("/", "_")

## drop rows if has nan values
data = data.dropna()

## reindex the data
data = data.reset_index(drop=True)

## print 0 variance columns
print(data.columns[data.var() == 0])

## check if any columns has zero variance and drop the columns
data = data.loc[:, data.var() != 0]

## add the Date column back to the data
data["Date"] = date

# convert "Date" column to datetime
data["Date"] = pd.to_datetime(data["Date"])

## for each parameter, check if the parameter changes over the years by plot boxplot for rach year
## get the unique years and sort them
years = data["Date"].dt.year.unique()
years.sort()

## get the unique parameters
parameters = data.columns
parameters = parameters[parameters != "Date"]

# for each parameter, check if the parameter changes over the years by plot boxplot for rach year
for parameter in parameters:
    ## create a list to store each year's data
    year_data = []

    for year in years:
        data_year = data[data["Date"].dt.year == year]
        data_year = data_year.drop("Date", axis=1)
        data_year = data_year[parameter]
        year_data.append(data_year)

    ## plot the boxplot
    plt.boxplot(year_data)

    ## use two digits for the year
    new_years = [str(year)[2:] for year in years]

    plt.xticks(range(1, len(new_years)+1), new_years)

    plt.title(parameter)
    plt.xlabel("Year")
    plt.ylabel(parameter)

    ## save data
    plt.savefig(f"/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/result/plots/variable_change_over_year/13_counties/{parameter}_13_counties.png", dpi=300)
    plt.show()

