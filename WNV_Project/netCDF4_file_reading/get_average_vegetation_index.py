import xarray as xr
import pandas as pd
import numpy as np
import glob

# read the netcdf4 files
ds = xr.open_dataset("/Users/ericliao/Desktop/WNV_project_files/weather_and_land_use/precipitation_monthly_2000_to_2021/leaf_index_1999_2020_data.nc")

## read the nonhuamn data from the csv file
human_neural_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/"
    "human_neuroinvasive_wnv_rate_log_population_correct_poverty.csv", index_col=0
)

## The latitude range is between 25.1 N and 49.4 N and the longitude range is between 66.9 W and 124.7 W
## access all variables from ds where time is all time and the area is belonged to U.S. Contiguous 48 United States

# create a data array based on the latitude from human_df
latitude_da = xr.DataArray(human_neural_df["County_Seat_Latitude"].values, dims="county")
# create a data array based on the longitude from human_df
longitude_da = xr.DataArray(human_neural_df["County_Seat_Longitude"].values, dims="county")
# create a data array based on the year from nonhuman_df
time_da = xr.DataArray(human_neural_df["Year"].values.astype(str).astype("datetime64[M]"), dims="county")

###### high leaf vegetation index ######
# # shift the whole data array 3 months earlier to get the lowest temperature in the previous October until this year August
shift_Oct_previous_year_lai_hv_mean = ds["lai_hv"].shift(time=3).coarsen(time=12, coord_func="max").mean()

# # get the lowest temperature in the previous October until this year August based on the location of the county
high_index_shift_values = (
    shift_Oct_previous_year_lai_hv_mean.sel(latitude=latitude_da, longitude=longitude_da, method="nearest")
).sel(time=time_da, method="bfill")

# # add the lowest temperature in the previous October until this year August to the human_df
human_neural_df["lai_hv_average_annual"] = high_index_shift_values.values
#########################

###### low leaf vegetation index ######
# # shift the whole data array 3 months earlier to get the lowest temperature in the previous October until this year August
shift_Oct_previous_year_lai_lv_mean = ds["lai_lv"].shift(time=3).coarsen(time=12, coord_func="max").mean()

# # get the lowest temperature in the previous October until this year August based on the location of the county
low_index_shift_values = (
    shift_Oct_previous_year_lai_lv_mean.sel(latitude=latitude_da, longitude=longitude_da, method="nearest")
).sel(time=time_da, method="bfill")

# # add the lowest temperature in the previous October until this year August to the human_df
human_neural_df["lai_lv_average_annual"] = low_index_shift_values.values
#########################

# save the human_df to a csv file
human_neural_df.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/"
                       "human_neuroinvasive_wnv_rate_log_population_correct_poverty_mean_green_index.csv", index=False)