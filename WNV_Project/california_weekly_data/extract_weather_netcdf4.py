import xarray as xr
import pandas as pd
import numpy as np
import glob

# create file list for all the netcdf4 files
# file_list = glob.glob("/Users/ericliao/Desktop/WNV_project_files/weather_and_land_use/precipitation_monthly_2000_to_2021/*.nc")
# # Read in the netCDF4 files
# ds = xr.open_mfdataset(file_list)

ds = xr.open_dataset("/Users/ericliao/Desktop/WNV_project_files/weather_and_land_use/precipitation_monthly_1999_to_2022/"
                     "land_monthly_1999_to_2022.nc")

# temp = ds["u10"]
print(ds)
## access ds u10 variable where time is all time and the area is belonged to U.S. Contiguous 48 United States
## (http://www.cohp.org/extremes/extreme_points.html)
## The latitude range is between 25.1 N and 49.4 N and the longitude range is between 66.9 W and 124.7 W
# alltime_u10 = ds["u10"].sel(longitude=slice(235.3, 293.1), latitude=slice(49.4, 25.1))
## different way to get the same result above
# alltime_u10 = ds.u10[:, ds.latitude > 25.1 and ds.latitude < 49.4 , ds.longitude > 235.3 and ds.longitude < 293.1]

## access ds u10 varaible at certain time and the area is belonged to U.S. Contiguous 48 United States
# certain_time_u10 = ds['u10'].sel(time='2012-01-01', longitude=slice(235.3, 293.1), latitude=slice(49.4, 25.1))

## access all variables from ds where time is all time and the area is belonged to U.S. Contiguous 48 United States
# alltime_all_var = ds.sel(longitude=slice(235, 246.5), latitude=slice(25.1, 49.4))

## read the nonhuamn data from the csv file
human_neural_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/disease_data_weekly_CA/final_file/"
    "cali_week_wnnd_wnf_multi_years.csv", index_col=0
)

## create a column to combine the year, month as two digits, and date 1st of every month
human_neural_df["Date"] = human_neural_df["Year"].astype(str) + "-" + human_neural_df["Month"].astype(str).str.zfill(2) + "-01"

# create a data array based on the latitude from human_df
latitude_da = xr.DataArray(human_neural_df["County_Seat_Latitude"].values, dims="county")
# create a data array based on the longitude from human_df
longitude_da = xr.DataArray(human_neural_df["County_Seat_Longitude"].values, dims="county")
# create a data array based on the date from nonhuman_df
time_da = xr.DataArray(human_neural_df["Date"].values.astype("datetime64[D]"), dims="county")

alltime_all_var_values = ds.sel(latitude=latitude_da, longitude=longitude_da, expver=1, method="nearest")
print(ds.sel(latitude=40.59515370528033, longitude=-74.60381627815474, expver=1, method="nearest")['u10'].values)


alltime_all_var_values_shift_1 = alltime_all_var_values.shift(time=1).sel(time=time_da, method="nearest")

human_neural_df["u10_1m_shift"] = alltime_all_var_values_shift_1["u10"].values
human_neural_df["v10_1m_shift"] = alltime_all_var_values_shift_1["v10"].values
human_neural_df["t2m_1m_shift"] = alltime_all_var_values_shift_1["t2m"].values
human_neural_df["lai_hv_1m_shift"] = alltime_all_var_values_shift_1["lai_hv"].values
human_neural_df["lai_lv_1m_shift"] = alltime_all_var_values_shift_1["lai_lv"].values
human_neural_df["src_1m_shift"] = alltime_all_var_values_shift_1["src"].values
human_neural_df["sf_1m_shift"] = alltime_all_var_values_shift_1["sf"].values
human_neural_df["sro_1m_shift"] = alltime_all_var_values_shift_1["sro"].values
human_neural_df["tp_1m_shift"] = alltime_all_var_values_shift_1["tp"].values

alltime_all_var_values_shift_2 = alltime_all_var_values.shift(time=2).sel(time=time_da, method="nearest")

human_neural_df["u10_2m_shift"] = alltime_all_var_values_shift_2["u10"].values
human_neural_df["v10_2m_shift"] = alltime_all_var_values_shift_2["v10"].values
human_neural_df["t2m_2m_shift"] = alltime_all_var_values_shift_2["t2m"].values
human_neural_df["lai_hv_2m_shift"] = alltime_all_var_values_shift_2["lai_hv"].values
human_neural_df["lai_lv_2m_shift"] = alltime_all_var_values_shift_2["lai_lv"].values
human_neural_df["src_2m_shift"] = alltime_all_var_values_shift_2["src"].values
human_neural_df["sf_2m_shift"] = alltime_all_var_values_shift_2["sf"].values
human_neural_df["sro_2m_shift"] = alltime_all_var_values_shift_2["sro"].values
human_neural_df["tp_2m_shift"] = alltime_all_var_values_shift_2["tp"].values

alltime_all_var_values_shift_12 = alltime_all_var_values.shift(time=12).sel(time=time_da, method="nearest")

human_neural_df["u10_12m_shift"] = alltime_all_var_values_shift_12["u10"].values
human_neural_df["v10_12m_shift"] = alltime_all_var_values_shift_12["v10"].values
human_neural_df["t2m_12m_shift"] = alltime_all_var_values_shift_12["t2m"].values
human_neural_df["lai_hv_12m_shift"] = alltime_all_var_values_shift_12["lai_hv"].values
human_neural_df["lai_lv_12m_shift"] = alltime_all_var_values_shift_12["lai_lv"].values
human_neural_df["src_12m_shift"] = alltime_all_var_values_shift_12["src"].values
human_neural_df["sf_12m_shift"] = alltime_all_var_values_shift_12["sf"].values
human_neural_df["sro_12m_shift"] = alltime_all_var_values_shift_12["sro"].values
human_neural_df["tp_12m_shift"] = alltime_all_var_values_shift_12["tp"].values

# alltime_all_var_values_shift_3 = alltime_all_var_values.shift(time=3).sel(time=time_da, method="nearest")
#
# human_neural_df["u10_3m_shift"] = alltime_all_var_values_shift_3["u10"].values
# human_neural_df["v10_3m_shift"] = alltime_all_var_values_shift_3["v10"].values
# human_neural_df["t2m_3m_shift"] = alltime_all_var_values_shift_3["t2m"].values
# human_neural_df["lai_hv_3m_shift"] = alltime_all_var_values_shift_3["lai_hv"].values
# human_neural_df["lai_lv_3m_shift"] = alltime_all_var_values_shift_3["lai_lv"].values
# human_neural_df["src_3m_shift"] = alltime_all_var_values_shift_3["src"].values
# human_neural_df["sf_3m_shift"] = alltime_all_var_values_shift_3["sf"].values
# human_neural_df["sro_3m_shift"] = alltime_all_var_values_shift_3["sro"].values
# human_neural_df["tp_3m_shift"] = alltime_all_var_values_shift_3["tp"].values
#
# alltime_all_var_values_shift_4 = alltime_all_var_values.shift(time=4).sel(time=time_da, method="nearest")
#
# human_neural_df["u10_4m_shift"] = alltime_all_var_values_shift_4["u10"].values
# human_neural_df["v10_4m_shift"] = alltime_all_var_values_shift_4["v10"].values
# human_neural_df["t2m_4m_shift"] = alltime_all_var_values_shift_4["t2m"].values
# human_neural_df["lai_hv_4m_shift"] = alltime_all_var_values_shift_4["lai_hv"].values
# human_neural_df["lai_lv_4m_shift"] = alltime_all_var_values_shift_4["lai_lv"].values
# human_neural_df["src_4m_shift"] = alltime_all_var_values_shift_4["src"].values
# human_neural_df["sf_4m_shift"] = alltime_all_var_values_shift_4["sf"].values
# human_neural_df["sro_4m_shift"] = alltime_all_var_values_shift_4["sro"].values
# human_neural_df["tp_4m_shift"] = alltime_all_var_values_shift_4["tp"].values


human_neural_df.to_csv(
    "/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/add_0_for_no_wnv/"
    "cali_week_wnnd_wnf_weather_shift.csv",
    index=False,
)

