import xarray as xr
import pandas as pd
import numpy as np
import glob
import scipy as sp

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
    "/Users/ericliao/Desktop/compensate.csv"
)


## interpolate the missing weather data from nearest neighbor
x_latitude = np.arange(0, alltime_all_var.latitude.size)
y_longitude = np.arange(0, alltime_all_var.longitude.size)
z_time = np.arange(0, alltime_all_var.time.size)

# mask invalid values
np_array = np.ma.masked_invalid(alltime_all_var['u10'])
zz_time, yy_longitude, xx_latitude = np.meshgrid(z_time, y_longitude, x_latitude, sparse=True)

# get only the valid values
x1_latitude = xx_latitude[~np_array.mask]
y1_longitude = yy_longitude[~np_array.mask]
z1_time = zz_time[~np_array.mask]
np_array_2 = np_array[~np_array.mask]

np_inperpolate = sp.interpolate.griddata((z1_time, x1_latitude, y1_longitude), np_array_2.ravel(), (zz_time, xx_latitude, yy_longitude), method='nearest')

# put the interpolate data back to the dataset
alltime_all_var['u10'] = np_inperpolate

# create a data array based on the latitude from human_df
latitude_da = xr.DataArray(human_neural_df["County_Centroid_Latitude"].values, dims="county")
# create a data array based on the longitude from human_df
longitude_da = xr.DataArray(human_neural_df["County_Centroid_Longitude"].values + 360, dims="county")
# create a data array based on the year from nonhuman_df

## todo: remember, the converting below, the date only has January 1st for every year (if need to use different month weather, need to change code)
time_da = xr.DataArray(human_neural_df["Year"].values.astype(int).astype(str).astype("datetime64[M]"), dims="county")
# retrieve data from alltime_all_var dataset nearest to the latitude and longitude data arrays

Jan_u10 = new_alltime_all_var.sel(latitude=latitude_da, longitude=longitude_da, time=time_da, method="nearest")


## todo: need to deal with the nan values in the weather data for certain area
human_neural_df["u10_Jan"] = new_alltime_all_var["u10"].values
# human_neural_df["v10_Jan"] = alltime_all_var_values["v10"].values
# human_neural_df["d2m_Jan"] = alltime_all_var_values["d2m"].values
# human_neural_df["t2m_Jan"] = alltime_all_var_values["t2m"].values
# human_neural_df["src_Jan"] = alltime_all_var_values["src"].values
# human_neural_df["skt_Jan"] = alltime_all_var_values["skt"].values
# human_neural_df["snowc_Jan"] = alltime_all_var_values["snowc"].values
# human_neural_df["rsn_Jan"] = alltime_all_var_values["rsn"].values
# human_neural_df["sde_Jan"] = alltime_all_var_values["sde"].values
# human_neural_df["sf_Jan"] = alltime_all_var_values["sf"].values
# human_neural_df["tsn_Jan"] = alltime_all_var_values["tsn"].values
# human_neural_df["tp_Jan"] = alltime_all_var_values["tp"].values
# human_neural_df["swvl1_Jan"] = alltime_all_var_values["swvl1"].values

human_neural_df.to_csv(
    "/Users/ericliao/Desktop/compensate.csv",
    index=False,
)

