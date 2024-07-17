import xarray as xr
import pandas as pd
import numpy as np
import glob

# create file list for all the netcdf4 files
ds = xr.open_dataset("/Users/ericliao/Desktop/WNV_project_files/weather_and_land_use/precipitation_monthly_1999_to_2022/"
                     "land_monthly_1999_to_2022.nc")

print(ds)
## access ds u10 variable where time is all time and the area is belonged to U.S. Contiguous 48 United States
## (http://www.cohp.org/extremes/extreme_points.html)
## The latitude range is between 25.1 N and 49.4 N and the longitude range is between 66.9 W and 124.7 W
## access all variables from ds where time is all time and the area is belonged to U.S. Contiguous 48 United States

## read the nonhuamn data from the csv file
cdc_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/cdc_sum_organism_binary.csv",
    index_col=0
)

## create a column to combine the year, month as two digits, and date 1st of every month
cdc_df["Date"] = cdc_df["Year"].astype(str) + "-" + cdc_df["Month"].astype(str).str.zfill(2) + "-01"

# create a data array based on the latitude from human_df
latitude_da = xr.DataArray(cdc_df["Latitude"].values, dims="county")
# create a data array based on the longitude from human_df
longitude_da = xr.DataArray(cdc_df["Longitude"].values, dims="county")
# create a data array based on the date from nonhuman_df
time_da = xr.DataArray(cdc_df["Date"].values.astype("datetime64[D]"), dims="county")

# create a list to store all the variable name
variable_list = ["u10", "v10", "t2m", "lai_hv", "lai_lv", "src", "sf", "sro", 'tp']
new_variable_name_list = ["u10_1m_shift", "v10_1m_shift", "t2m_1m_shift", "lai_hv_1m_shift", "lai_lv_1m_shift",
                          "src_1m_shift", "sf_1m_shift", "sro_1m_shift", 'tp_1m_shift']

# select each variable separately
for variable, new_variable in zip(variable_list, new_variable_name_list):
    alltime_all_var_values = ds[variable].sel(latitude=latitude_da, longitude=longitude_da, expver=1, method="nearest")
    # print(ds.sel(latitude=40.59515370528033, longitude=-74.60381627815474, expver=1, method="nearest")['u10'].values)

    alltime_all_var_values_shift_1 = alltime_all_var_values.shift(time=1).sel(time=time_da, method="nearest")

    cdc_df[new_variable] = alltime_all_var_values_shift_1.values
    alltime_all_var_values.close()
    print("finish ", variable)

cdc_df.to_csv(
    "/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/cdc_sum_organism_climate_binary.csv",
    index=False,
)

