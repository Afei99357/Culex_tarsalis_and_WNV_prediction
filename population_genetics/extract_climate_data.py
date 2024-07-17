import cv2
import xarray as xr
import numpy as np
import pandas as pd

# read disease data file
mos_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/Ctarsalis_sample_w_GPS_climate.csv",
    sep=",",
)

# create file list for all the netcdf4 files
ds = xr.open_dataset(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/climate_data/land_monthly_climate_2009_2012_partial_canada.nc"
)

print(ds.variables.mapping.keys())

# create a data array based on the latitude from mos_df
latitude_da = xr.DataArray(mos_df["GPS.Lat"].values, dims="location")
# create a data array based on the longitude from mos_df
longitude_da = xr.DataArray(mos_df["GPS.Lon"].values, dims="location")
# create a data array based on the time from mos_df
time_da = xr.DataArray(mos_df["date"].values, dims="location")

# create a list to store all the variable name
variable_list = ["lai_hv", "lai_lv", "src", "sro"]

# select each variable separately
for variable in variable_list:
    alltime_all_var_values = ds[variable].sel(
        latitude=latitude_da, longitude=longitude_da, time=time_da, method="nearest"
    )
    # print(ds.sel(latitude=40.59515370528033, longitude=-74.60381627815474, expver=1, method="nearest")['u10'].values)

    alltime_all_var_values = alltime_all_var_values.sel(time=time_da, method="nearest")

    mos_df[variable] = alltime_all_var_values.values
    alltime_all_var_values.close()
    print("finish ", variable)

# save the dataframe as a csv file
mos_df.to_csv(
    "/Users/ericliao/Desktop/WNV_project_files/Ctarsalis_sample_w_GPS_climate_new.csv",
    index=False,
)
