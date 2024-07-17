import xarray as xr
import pandas as pd

# read the netcdf4 files
ds = xr.open_dataset("/Users/ericliao/Desktop/WNV_project_files/weather_and_land_use/precipitation_monthly_2000_to_2021/leaf_index_2000_2020_data.nc")

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

month_list = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
for i, month in zip(range(12), month_list):
    ## todo: remember, the converting below, the date only has January 1st for every year (if need to use different month weather, need to change code)
    time_da = xr.DataArray(human_neural_df["Year"].values.astype(str).astype("datetime64[M]") + i, dims="county")
    # retrieve data from alltime_all_var dataset nearest to the latitude and longitude data arrays
    alltime_all_var_values = ds.sel(latitude=latitude_da, longitude=longitude_da, time=time_da, method="nearest")

    human_neural_df["lai_hv_" + month] = alltime_all_var_values["lai_hv"].values
    human_neural_df["lai_lv_" + month] = alltime_all_var_values["lai_lv"].values


human_neural_df.to_csv(
    "/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/"
    "human_neuroinvasive_wnv_rate_log_population_correct_poverty_lai.csv",
    index=False,
)
