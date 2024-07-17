import xarray as xr
import pandas as pd

# read disease data file
mos_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/"
    "Ctarsalis_sample_w_GPS_land_use.csv",
    sep=",",
)

# create file list for all the netcdf4 files
ds = xr.open_dataset(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/climate_data/"
    "land_monthly_climate_2009_2012_partial_canada.nc"
)

# create a data array based on the latitude from mos_df
latitude_da = xr.DataArray(mos_df["GPS.Lat"].values, dims="county")
# create a data array based on the longitude from mos_df
longitude_da = xr.DataArray(mos_df["GPS.Lon"].values, dims="county")
# create a data array based on the time from mos_df
# time_da = xr.DataArray(mos_df["date"].values, dims="county")

# create a list to store all the variable name
variable_list = [
    "u10",
    "v10",
    "t2m",
    "evabs",
    "lai_hv",
    "lai_lv",
    "src",
    "sf",
    "ssr",
    "sro",
    "e",
    "tp",
    "swvl1",
]

# select each variable separately
for variable in variable_list:
    alltime_all_var_values = ds[variable].sel(
        latitude=latitude_da, longitude=longitude_da, method="nearest"
    )
    alltime_all_var_values = alltime_all_var_values.sortby("time")
    # print(ds.sel(latitude=40.59515370528033, longitude=-74.60381627815474, expver=1, method="nearest")['u10'].values)
    # get the average value of year
    average = alltime_all_var_values.sel(time=slice("2012-01-01", "2012-12-31")).mean(
        dim="time"
    )
    # min_value = alltime_all_var_values.sel(time=slice("2012-01-01", "2012-12-31")).min(dim="time")
    # max_value = alltime_all_var_values.sel(time=slice("2012-01-01", "2012-12-31")).max(dim="time")

    mos_df["avg_" + variable] = average.values
    # mos_df['min_' + variable] = min_value.values
    # mos_df['max_' + variable] = max_value.values
    alltime_all_var_values.close()
    print("finish ", variable)

# save the dataframe as a csv file
mos_df.to_csv(
    "/Users/ericliao/Desktop/WNV_project_files/Ctarsalis_sample_w_GPS_climate_average_new.csv",
    index=False,
)
