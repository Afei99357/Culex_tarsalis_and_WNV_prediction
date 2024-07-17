import xarray as xr
import pandas as pd
import numpy as np

# read disease data file
eur_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/WNV/Europe_data/adding_0_case/europe_data_with_coordinates_landuse_0_case.csv",
    index_col=0
)

# create file list for all the netcdf4 files
ds = xr.open_dataset(
    "/Users/ericliao/Desktop/WNV_project_files/WNV/Europe_data/land_monthly_2007_2023_europe.nc"
)

# create a data array based on the latitude
latitude_da = xr.DataArray(eur_df["Latitude"].values, dims="region")
# create a data array based on the longitude
longitude_da = xr.DataArray(eur_df["Longitude"].values, dims="region")

year_da = xr.DataArray(eur_df["Time"].values, dims="region")

# create a list to store all the variable name
variable_list = [
    "u10",
    "v10",
    "t2m",
    "lai_hv",
    "lai_lv",
    "src",
    "sf",
    "smlt",
    "ssr",
    "sro",
    "e",
    "tp",
    "swvl1",
]

ds = ds.sel(expver=1, method="nearest")

alltime_all_var_values_avg = ds.groupby('time.year').mean(dim="time")

# select each variable separately
for variable in variable_list:
    print("start ", variable)
    alltime_all_var_values_variable = alltime_all_var_values_avg.sel(
        latitude=latitude_da, longitude=longitude_da, method="nearest"
    )
    # assign the average value of the year to the dataframe
    eur_df["avg_" + variable] = alltime_all_var_values_variable[variable].sel(year=year_da).values
    print("finish ", variable)

# save the dataframe as a csv file
eur_df.to_csv(
    "/Users/ericliao/Desktop/WNV_project_files/WNV/Europe_data/adding_0_case/europe_data_with_coordinates_landuse_climate_0_case.csv"
)
