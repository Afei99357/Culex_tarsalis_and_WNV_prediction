import xarray as xr
import pandas as pd
import numpy as np
import glob

# create file list for all the netcdf4 files
file_list = glob.glob("/Users/ericliao/Desktop/WNV_project_files/weather_and_land_use/precipitation_monthly_2000_to_2021/*.nc")
# Read in the netCDF4 files
ds = xr.open_mfdataset(file_list)

temp = ds["u10"]
print(ds)
## access ds u10 variable where time is all time and the area is belonged to U.S. Contiguous 48 United States
## (http://www.cohp.org/extremes/extreme_points.html)
## The latitude range is between 25.1 N and 49.4 N and the longitude range is between 66.9 W and 124.7 W
alltime_u10 = ds["u10"].sel(longitude=slice(235.3, 293.1), latitude=slice(49.4, 25.1))
## different way to get the same result above
# alltime_u10 = ds.u10[:, ds.latitude > 25.1 and ds.latitude < 49.4 , ds.longitude > 235.3 and ds.longitude < 293.1]

## access ds u10 varaible at certain time and the area is belonged to U.S. Contiguous 48 United States
# certain_time_u10 = ds['u10'].sel(time='2012-01-01', longitude=slice(235.3, 293.1), latitude=slice(49.4, 25.1))

## access all variables from ds where time is all time and the area is belonged to U.S. Contiguous 48 United States
alltime_all_var = ds.sel(longitude=slice(235.3, 293.1), latitude=slice(49.4, 25.1)
)

## read the nonhuamn data from the csv file
human_neural_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_13_to_2023_02_20/dataset/human_neuroinvasive_with_weather_with_county_seat_modify.csv", index_col=False
)

# create a data array based on the latitude from human_df
latitude_da = xr.DataArray(human_neural_df["County_Seat_Latitude"].values, dims="county")
# create a data array based on the longitude from human_df
longitude_da = xr.DataArray(human_neural_df["County_Seat_Longitude"].values + 360, dims="county")
# create a data array based on the year from nonhuman_df
time_da = xr.DataArray(human_neural_df["Year"].values.astype(str).astype("datetime64[M]"), dims="county")

## need to remove
alltime_all_var["d2m"].sel(latitude=41.28348123, longitude=360 - 70.10141255, method="nearest").values

###### temperature ######
# # shift the whole data array 3 months earlier to get the lowest temperature in the previous October until this year August
shift_Oct_previous_year_d2m_lowest = alltime_all_var["d2m"].shift(time=3).coarsen(time=12, coord_func="max").min()

# # get the lowest temperature in the previous October until this year August based on the location of the county
d2m_lowest_shift_values = (
    shift_Oct_previous_year_d2m_lowest.sel(latitude=latitude_da, longitude=longitude_da, method="nearest")
).sel(time=time_da, method="bfill")

# # add the lowest temperature in the previous October until this year August to the human_df
human_neural_df["d2m_min_Oct_to_Aug"] = d2m_lowest_shift_values.values
#########################

###### snow cover ######
## for the maximum snow cover in the previous October until this year August
# # shift the whole data array 3 months earlier to get the maximum snow cover in the previous October until this year August
shift_Oct_previous_year_snowcover_max = alltime_all_var["snowc"].shift(time=3).coarsen(time=12, coord_func="max").max()

# # get the highest snow cover in the previous October until this year August based on the location of the county
snowc_max_shift_values = (
    shift_Oct_previous_year_snowcover_max.sel(latitude=latitude_da, longitude=longitude_da, method="nearest")
).sel(time=time_da, method="bfill")

# # add the lowest temperature in the previous October until this year August to the human_df
human_neural_df["snowc_max_Oct_to_Aug"] = snowc_max_shift_values.values
#########################

###### snow fall ######
## for the accumulated snow fall in the previous October until this year August
# # shift the whole data array 3 months earlier to get the accumulated snow fall in the previous October until this year August
shift_Oct_previous_year_snowfall_acc = alltime_all_var["sf"].shift(time=3).coarsen(time=12, coord_func="max").sum()

# # get the accumulated snow fall in the previous October until this year August based on the location of the county
sf_acc_shift_values = (
    shift_Oct_previous_year_snowfall_acc.sel(latitude=latitude_da, longitude=longitude_da, method="nearest")
).sel(time=time_da, method="bfill")

# # add the accumulated snow fall in the previous October until this year August to the human_df
human_neural_df["sf_acc_Oct_to_Aug"] = sf_acc_shift_values.values
#########################

###### snow depth ######
## for the maximum snow depth in the previous October until this year August
# # shift the whole data array 3 months earlier to get the maximum snow depth in the previous October until this year August
shift_Oct_previous_year_snowdepth_max = alltime_all_var["sde"].shift(time=3).coarsen(time=12, coord_func="max").max()

# # get the maximum snow depth in the previous October until this year August based on the location of the county
sde_max_shift_values = (
    shift_Oct_previous_year_snowdepth_max.sel(latitude=latitude_da, longitude=longitude_da, method="nearest")
).sel(time=time_da, method="bfill")

# # add the maximum snow depth in the previous October until this year August to the human_df
human_neural_df["sde_max_Oct_to_Aug"] = sde_max_shift_values.values
#########################

###### snow layer temperature ######
## for the lowest snow layer temperature in the previous October until this year August
# # shift the whole data array 3 months earlier to get the lowest snow layer temperature in the previous October until this year August
shift_Oct_previous_year_slt_min = alltime_all_var["tsn"].shift(time=3).coarsen(time=12, coord_func="max").min()

# # get the lowest snow layer temperature in the previous October until this year August based on the location of the county
tsn_min_shift_values = (
    shift_Oct_previous_year_slt_min.sel(latitude=latitude_da, longitude=longitude_da, method="nearest")
).sel(time=time_da, method="bfill")

# # add the lowest snow layer temperature in the previous October until this year August to the human_df
human_neural_df["tsn_min_Oct_to_Aug"] = tsn_min_shift_values.values
#########################

###### precipitation ######
## for the accumulated precipitation in the previous October until this year August
# # shift the whole data array 3 months earlier to get the accumulated precipitation in the previous October until this year August
shift_Oct_previous_year_tp_acc = alltime_all_var["tp"].shift(time=3).coarsen(time=12, coord_func="max").sum()

# # get the accumulated precipitation in the previous October until this year August based on the location of the county
tp_acc_shift_values = (
    shift_Oct_previous_year_tp_acc.sel(latitude=latitude_da, longitude=longitude_da, method="nearest")
).sel(time=time_da, method="bfill")

# # add the accumulated precipitation in the previous October until this year August to the human_df
human_neural_df["tp_acc_Oct_to_Aug"] = tp_acc_shift_values.values
#########################

human_neural_df.to_csv(
    "/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_13_to_2023_02_20/dataset/human_neuroinvasive_with_extreme_weather_with_county_seat_modify.csv",
    index=False,
)

