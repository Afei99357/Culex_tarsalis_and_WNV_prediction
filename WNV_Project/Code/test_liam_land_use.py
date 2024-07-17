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
mos_df = pd.read_csv("/Users/ericliao/Downloads/data.csv")

# create a data array based on the latitude from human_df
latitude_da = xr.DataArray(mos_df["Latitude"].values, dims="county")
# create a data array based on the longitude from human_df
longitude_da = xr.DataArray(mos_df["Longitude"].values, dims="county")

# create a list of land use types
land_use_list = ["Evergreen/Deciduous Needleleaf Trees", "Evergreen Broadleaf Trees", "Deciduous Broadleaf Trees",
                 "Mixed Trees", "Shrub", "Herbaceous", "Culture/Managed", "Wetland", "Urban/Built", "Snow/Ice",
                 "Barren", "Water"]

# create a for loop to loop through the land use list, and assign th e values to mos_df as new column
for land_use in land_use_list:
    land_use_da = dataset[land_use].sel(latitude=latitude_da, longitude=longitude_da, method='nearest')
    mos_df[land_use] = land_use_da.values

# save the mos_df to csv file
mos_df.to_csv("/Users/ericliao/Downloads/data_land_use_liam.csv")