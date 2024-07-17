import pandas as pd
import xarray as xr
import cv2
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

## read FIPS code info from old files
fips_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/data/cali_week_wnnd.csv", sep=",")


## get unique county, FIPS, Latitude, Longgitude pairs
fips_df = fips_df[["County", "FIPS", "Latitude", "Longitude", "Avian Phylodiversity"]].drop_duplicates()

## adding FIPS code to the new data
data_california_new = pd.merge(data_california_new, fips_df, how="left", on="County")

## check any missing values in the FIPS column
print(data_california_new[data_california_new["FIPS"].isna()])

## reorder all the columns as Year, Month, County, FIPS, Cases
data = data_california_new[["Year", "Month", "County", "FIPS", "Latitude", "Longitude", "Cases", "Avian Phylodiversity"]]

## adding population
df_population = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/yearly/disease_human_neuroinvasive_whole_year.csv", sep=",")

## get the population of California in 2020
df_population = df_population[df_population["State"] == "California"]

## get county and population in 2020
df_population = df_population[["County", "Population", "Year"]].drop_duplicates()

## get the population of California in 2020
df_population = df_population[df_population["Year"] == 2020]

## drop the year column
df_population = df_population.drop("Year", axis=1)

## lower case the county column
df_population["County"] = df_population["County"].str.lower()
## remove the leading and trailing white spaces
df_population["County"] = df_population["County"].str.strip()

## merge the population data with the original data
data = pd.merge(data, df_population, how="left", on="County")

## rename the "Cases" column to "Human_Disease_Count"
data = data.rename(columns={"Cases": "Human_Disease_Count"})

####### impute 0 cases
## import CDC national data
df_cdc = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/monthly/combine_cdc_all_environmental_variable_all_2024.csv", sep=",")

## get only California data
df_cdc = df_cdc[df_cdc["State"] == "california"]

## keep columns as "Total_Bird_WNV_Count", "Mos_WNV_Count", "Horse_WNV_Count", Year, Month, County
df_cdc = df_cdc[["Total_Bird_WNV_Count", "Mos_WNV_Count", "Horse_WNV_Count", "Year", "Month", "County"]]

## merge the CDC data with the original data
data = pd.merge(data, df_cdc, how="left", on=["Year", "Month", "County"])

## for each row in the df, if "Human_WNND_Count", "Mos_WNV_Count"and "TotalBird_WNV_Count" are nan, fill 0 with "Human_Disease_Count"
data.loc[data["Human_Disease_Count"].isna() & data["Mos_WNV_Count"].isna() & data["Total_Bird_WNV_Count"].isna() & data["Horse_WNV_Count"], "Human_Disease_Count"] = 0

# ## adding an column for "average_human_case_monthly" by grouping the data by "FIPS" and 'Year' and calculate the mean of total same month of all years "Reported_human_cases"
# data["average_human_case_monthly"] = data.groupby(["FIPS", "Month"])["Human_Disease_Count"].transform("sum") / len(data["Year"].unique())
#
# ## for each row, if the "Human_WNND_Count" is nan, fill "Human_WNND_Count" with "average_human_case_monthly"
# data.loc[data["Human_Disease_Count"].isna(), "Human_Disease_Count"] = data["average_human_case_monthly"]

print(data["Human_Disease_Count"].isna().sum() / data.shape[0])

############# adding El Nino/La Nina data ######
print("start adding El Nino/La Nina data")
## read the El Nino/La Nina data
df_enso = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/Historical El Nino or La Nina episodes (1950-present)/Historical_El_Nino_or_La_Nina_episodes_1950_present.csv", sep=",")

## adding a new column to data to store the El Nino/La Nina data
data["ONI"] = ""

## based on year and month to add the El Nino/La Nina data to a list
enso_list = []
month_list = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
for index, row in data.iterrows():
    year = row["Year"]
    month = month_list[row["Month"]-1]
    ## based on the month to locate the column and year to locate the row, then add value to the list
    enso_list.append(df_enso.loc[df_enso["Year"] == year, str(month)].values[0])

## add the enso_list to the data
data["ONI"] = enso_list

## print finish adding El Nino/La Nina data
print("finish adding El Nino/La Nina data")

########### adding lan use ##############
# create a data array based on the latitude from human_df
latitude_da = xr.DataArray(data["Latitude"].values, dims="county")
# create a data array based on the longitude from human_df
longitude_da = xr.DataArray(data["Longitude"].values, dims="county")

############# adding land use ########################
print("start adding land use")
## read the land use data
images = [
    f"/Users/ericliao/Desktop/WNV_project_files/WNV/climate/consensus_land_cover_data/"
    f"consensus_full_class_{i}.tif"
    for i in range(1, 13)
]

images = [cv2.imread(i)[:, :, 0].copy() for i in images]

# Let's add coordinates to the image and put it in an xarray DataArray
# The coordinates are latitude and longitude, where latitude spans from 90 degrees north to 56 degrees south,
# and longitude spans from 180 west to 180 east
# The resolution is inferred from the image shape
dataset = [
    xr.DataArray(
        im,
        coords=[np.linspace(90, -56, im.shape[0]), np.linspace(-180, 180, im.shape[1])],
        dims=["latitude", "longitude"],
    )
    for im in images
]

dataset = xr.Dataset(
    data_vars=dict(
        zip(
            [
                "Evergreen/Deciduous Needleleaf Trees",
                "Evergreen Broadleaf Trees",
                "Deciduous Broadleaf Trees",
                "Mixed Trees",
                "Shrub",
                "Herbaceous",
                "Culture/Managed",
                "Wetland",
                "Urban/Built",
                "Snow/Ice",
                "Barren",
                "Water",
            ],
            dataset,
        )
    )
)


# create a list of land use types
land_use_list = [
    "Evergreen/Deciduous Needleleaf Trees",
    "Evergreen Broadleaf Trees",
    "Deciduous Broadleaf Trees",
    "Mixed Trees",
    "Shrub",
    "Herbaceous",
    "Culture/Managed",
    "Wetland",
    "Urban/Built",
    "Snow/Ice",
    "Barren",
    "Water",
]

# create a for loop to loop through the land use list, and assign th e values to mos_df as new column
for land_use in land_use_list:
    land_use_da = dataset[land_use].sel(
        latitude=latitude_da, longitude=longitude_da, method="nearest"
    )
    data[land_use] = land_use_da.values

# close the dataset
dataset.close()

## print finish adding land use
print("finish adding land use")



################### extract the climate data ############################
## adding a Date column to the data build from the Year, Month, day use the first day of the month
data["Date"] = pd.to_datetime(data[["Year", "Month"]].assign(day=1))

# create a data array based on the time from mos_df
# create a data array based on the date from nonhuman_df
time_da = xr.DataArray(data["Date"].values.astype("datetime64[D]"), dims="county")

# create file list for all the netcdf4 files
ds = xr.open_dataset("/Users/ericliao/Desktop/WNV_project_files/WNV/climate/new_land_monthly_data_from_1999_to_2024_02.nc")

## sort the ds based on time
ds = ds.sortby("time")

# create a list to store all the variable name
variable_list = ["u10", "v10", "t2m", "lai_hv", "lai_lv", "src", "sf", "sro", 'tp']
new_variable_name_list = ["u10_1m_shift", "v10_1m_shift", "t2m_1m_shift", "lai_hv_1m_shift", "lai_lv_1m_shift",
                          "src_1m_shift", "sf_1m_shift", "sro_1m_shift", 'tp_1m_shift']

# slect each variable separately
for variable, new_variable in zip(variable_list, new_variable_name_list):
    alltime_all_var_values = ds[variable].sel(latitude=latitude_da, longitude=longitude_da, expver=1, method="nearest")

    alltime_all_var_values_shift_1 = alltime_all_var_values.shift(time=1).sel(time=time_da, method="nearest")

    data[new_variable] = alltime_all_var_values_shift_1.values
    alltime_all_var_values.close()
    print("finish ", variable)

# close the dataset
ds.close()

## ## remove , in population column
data["Population"] = data["Population"].str.replace(",", "")

## convert the population column to numeric
data["Population"] = pd.to_numeric(data["Population"])

##output the data
data.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/CA_human_data_2004_to_2023_final_new.csv", index=False)
# data.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/CA_human_data_2004_to_2023_final_impute_entire_04_23.csv", index=False)






