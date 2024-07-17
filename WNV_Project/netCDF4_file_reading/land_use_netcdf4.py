import xarray as xr
import pandas as pd
import numpy as np
import glob

# create file list for all the netcdf4 files
file_list = glob.glob("/Users/ericliao/Desktop/WNV_project_files/weather_and_land_use/land_cover_monthly_2000_to_2020/*.nc")

ds_1 = xr.open_dataset(file_list[0])


#### read the netcdf4 files and concatenate all the us part into one dataset

## read the nonhuamn data from the csv file
human_neural_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/disease_human_neuroinvasive_wnv_2000-2021_bird_demographic"
    "_land_area_weather_01_30_2023.csv"
)

# create a data array based on the latitude from human_df
latitude_da = xr.DataArray(human_neural_df["latitude"].values, dims="county")
# create a data array based on the longitude from human_df
longitude_da = xr.DataArray(human_neural_df["longitude"].values, dims="county")
# create a data array based on the time from human_df
time_da = xr.DataArray(human_neural_df["year"].values.astype(str).astype("datetime64[M]"), dims="county")

## ## get change count data
## The latitude range is between 25.1 N and 49.4 N and the longitude range is between 66.9 W and 124.7 W
ds_change_count = xr.concat([
    xr.open_dataset(file)
    .sel(lon=slice(-124.7, -66.9), lat=slice(49.4, 25.1))[['change_count']]
    .copy(deep=True)
    for file in file_list
], dim="time")
# sort ds based on time in ascending order
ds_change_count = ds_change_count.sortby("time")

alltime_change_counts = ds_change_count.sel(lat=latitude_da, lon=longitude_da, time=time_da, method="nearest")

human_neural_df["land_change_count_since_1992"] = alltime_change_counts["change_count"].values

## ## get lccs_class data
## The latitude range is between 25.1 N and 49.4 N and the longitude range is between 66.9 W and 124.7 W
ds_lccs_class = xr.concat([
    xr.open_dataset(file)
    .sel(lon=slice(-124.7, -66.9), lat=slice(49.4, 25.1))[['lccs_class']]
    .copy(deep=True)
    for file in file_list
], dim="time")
# sort ds based on time in ascending order
ds_lccs_class = ds_lccs_class.sortby("time")

alltime_lccs_class = ds_lccs_class.sel(lat=latitude_da, lon=longitude_da, time=time_da, method="nearest")

human_neural_df["land_use_class"] = alltime_lccs_class["lccs_class"].values

## ## get processed_flag data
## The latitude range is between 25.1 N and 49.4 N and the longitude range is between 66.9 W and 124.7 W
ds_processed_flag = xr.concat([
    xr.open_dataset(file)
    .sel(lon=slice(-124.7, -66.9), lat=slice(49.4, 25.1))[['processed_flag']]
    .copy(deep=True)
    for file in file_list
], dim="time")
# sort ds based on time in ascending order
ds_processed_flag = ds_processed_flag.sortby("time")

alltime_processed_flag = ds_processed_flag.sel(lat=latitude_da, lon=longitude_da, time=time_da, method="nearest")

human_neural_df["processed_flag_land_use"] = alltime_processed_flag["processed_flag"].values

# writer human_neural_df to csv file
human_neural_df.to_csv('/Users/ericliao/Desktop/disease_human_neuroinvasive_wnv_2000-2021_bird_demographic_land_cover.csv', index=False)

