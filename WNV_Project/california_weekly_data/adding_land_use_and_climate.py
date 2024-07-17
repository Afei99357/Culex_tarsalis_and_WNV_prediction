import cv2
import xarray as xr
import numpy as np
import pandas as pd


# Copy is to avoid keeping the other two colors in memory
images = [f'/Users/ericliao/Desktop/WNV_project_files/weather_and_land_use/consensus_land_cover_data/' \
          f'consensus_full_class_{i}.tif' for i in range(1, 13)]

images = [cv2.imread(i)[:, :, 0].copy() for i in images]

# Let's add coordinates to the image and put it in an xarray DataArray
# The coordinates are latitude and longitude, where latitude spans from 90 degrees north to 56 degrees south,
# and longitude spans from 180 west to 180 east
# The resolution is inferred from the image shape
dataset = [
    xr.DataArray(im, coords=[np.linspace(90, -56, im.shape[0]), np.linspace(-180, 180, im.shape[1])],
                 dims=['latitude', 'longitude'])
    for im in images
]

dataset = xr.Dataset(data_vars=dict(zip(
    [
        # Copilot had memorized this list
        "Evergreen/Deciduous Needleleaf Trees",
        "Evergreen Broadleaf Trees",
        "Deciduous Broadleaf Trees",
        "Mixed Trees",
        "Shrub",
        "Herbaceous",
        "Culture/Managed",
        "Wetland",
        "Urban/Built",
        "Snow/Ice",
        "Barren",
        "Water"
    ],
    dataset
)))

# read disease data file
mos_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
                 "add_0_for_no_wnv/cali_week_wnnd_human.csv", index_col=0)


# create a data array based on the latitude from human_df
latitude_da = xr.DataArray(mos_df["County_Seat_Latitude"].values, dims="county")
# create a data array based on the longitude from human_df
longitude_da = xr.DataArray(mos_df["County_Seat_Longitude"].values, dims="county")

# convert the date column to datetime format
mos_df["Date"] = pd.to_datetime(mos_df["Date"])
# create a data array based on the time from mos_df
time_da = xr.DataArray(mos_df["Date"].values, dims="county")

# create a list of land use types
land_use_list = ["Evergreen/Deciduous Needleleaf Trees", "Evergreen Broadleaf Trees", "Deciduous Broadleaf Trees",
                 "Mixed Trees", "Shrub", "Herbaceous", "Culture/Managed", "Wetland", "Urban/Built", "Snow/Ice",
                 "Barren", "Water"]

# create a for loop to loop through the land use list, and assign th e values to mos_df as new column
for land_use in land_use_list:
    land_use_da = dataset[land_use].sel(latitude=latitude_da, longitude=longitude_da, method='nearest')
    mos_df[land_use] = land_use_da.values

# close the dataset
dataset.close()

### extract the climate data
# create file list for all the netcdf4 files
ds = xr.open_dataset("/Users/ericliao/Desktop/WNV_project_files/weather_and_land_use/precipitation_monthly_1999_to_2022/"
                     "land_monthly_1999_to_2022.nc")

# create a list to store all the variable name
variable_list = ["u10", "v10", "t2m", "lai_hv", "lai_lv", "src", "sf", "sro", 'tp']
new_variable_name_list = ["u10_1m_shift", "v10_1m_shift", "t2m_1m_shift", "lai_hv_1m_shift", "lai_lv_1m_shift",
                          "src_1m_shift", "sf_1m_shift", "sro_1m_shift", 'tp_1m_shift']

# slect each variable separately
for variable, new_variable in zip(variable_list, new_variable_name_list):
    alltime_all_var_values = ds[variable].sel(latitude=latitude_da, longitude=longitude_da, expver=1, method="nearest")

    alltime_all_var_values_shift_1 = alltime_all_var_values.shift(time=1).sel(time=time_da, method="nearest")

    mos_df[new_variable] = alltime_all_var_values_shift_1.values
    alltime_all_var_values.close()
    print("finish ", variable)

# save the mos_df as csv file
mos_df.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
              "add_0_for_no_wnv/cali_week_wnnd_human_all_features.csv", index=False)



