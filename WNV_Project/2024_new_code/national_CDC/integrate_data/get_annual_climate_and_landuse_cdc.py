import pandas as pd
import xarray as xr
import cv2
import numpy as np


######### adding phylogenetic diversity index to the cdc dataset  ############
# load data
cdc_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/cdc_human_1999_to_2023/"
    "WNV_human_and_non-human_annual_by_county_2004_to_2023_impute_missing.csv",
    index_col=False,
    header=0
)

## drop the nan values inn column 'Logitude' and 'Latitude'
cdc_df = cdc_df.dropna(subset=["Longitude", "Latitude"])

## reindex the dataframe
cdc_df = cdc_df.reset_index(drop=True)

## create a Date column, value is the year-01-01
cdc_df["Date"] = cdc_df["Year"].astype(str) + "-01-01"

## convert the Date column to datetime
cdc_df["Date"] = pd.to_datetime(cdc_df["Date"])

################ adding climate variables to the cdc dataset   ####################
print("start adding climate variables")

ds = xr.open_dataset(
    "/Users/ericliao/Desktop/WNV_project_files/WNV/climate/new_land_monthly_data_from_1999_to_2024_02.nc"
)

## sort the ds based on time
ds = ds.sortby("time")

print(ds)
## access ds u10 variable where time is all time and the area is belonged to U.S. Contiguous 48 United States
## (http://www.cohp.org/extremes/extreme_points.html)
## The latitude range is between 25.1 N and 49.4 N and the longitude range is between 66.9 W and 124.7 W
## access all variables from ds where time is all time and the area is belonged to U.S. Contiguous 48 United States

# create a data array based on the latitude from human_df
latitude_da = xr.DataArray(cdc_df["Latitude"].values, dims="county")
# create a data array based on the longitude from human_df
longitude_da = xr.DataArray(cdc_df["Longitude"].values, dims="county")
# create a data array based on the Year from human_df, to select the climate data by annual average
time_da = xr.DataArray(cdc_df["Date"].values, dims="county")
year_da = xr.DataArray(cdc_df["Year"].values, dims="county")

# create a list to store all the variable name
variable_list = [
    "u10",
    "v10",
    "t2m",
    "lai_hv",
    "lai_lv",
    "src",
    "sf",
    "ssr",
    "sro",
    "e",
    "tp",
    "swvl1"
]

## reference: https://ncar.github.io/esds/posts/2021/yearly-averages-xarray/
def weighted_temporal_mean(ds, var):
    """
    weight by days in each month
    """
    # Determine the month length
    month_length = ds.time.dt.days_in_month
    # Calculate the weights
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()
    # Make sure the weights in each year add up to 1
    np.testing.assert_allclose(wgts.groupby("time.year").sum(xr.ALL_DIMS), 1.0)
    # Subset our dataset for our variable
    obs = ds[var]
    # Setup our masking for nan values
    cond = obs.isnull()
    ones = xr.where(cond, 0.0, 1.0)
    # Calculate the numerator
    obs_sum = (obs * wgts).resample(time="AS").sum(dim="time")
    # Calculate the denominator
    ones_out = (ones * wgts).resample(time="AS").sum(dim="time")
    # Return the weighted average
    return obs_sum / ones_out

### select each variable separately
for variable in variable_list:
    print("start ", variable)
    annual_variable = weighted_temporal_mean(ds, variable)
    average_variable = annual_variable.sel(latitude=latitude_da, longitude=longitude_da, time=time_da, expver=1, method="nearest")
    cdc_df["avg_" + variable] = average_variable.values
    average_variable.close()
    print("finish ", variable)

## get the max and min value of each variable
for variable in variable_list:
    print("start ", variable)
    ## group by year and get the max and min value of each variable
    max_variable = ds[variable].groupby("time.year").max(dim="time")
    min_variable = ds[variable].groupby("time.year").min(dim="time")

    ## select the max and min value of each variable
    max_variable = max_variable.sel(latitude=latitude_da, longitude=longitude_da, year=year_da, expver=1, method="nearest")
    min_variable = min_variable.sel(latitude=latitude_da, longitude=longitude_da, year=year_da, expver=1, method="nearest")

    ## assign the max and min value to cdc_df as new column
    cdc_df["max_" + variable] = max_variable.values
    cdc_df["min_" + variable] = min_variable.values
    max_variable.close()
    min_variable.close()
    print("finish ", variable)

############# adding land use ########################
print("start adding land use")
## read the land use data
images = [
    f"/Users/ericliao/Desktop/WNV_project_files/WNV/climate/consensus_land_cover_data/"
    f"consensus_full_class_{i}.tif"
    for i in range(1, 13)
]

images = [cv2.imread(i)[:, :, 0].copy() for i in images]

# Let's add coordinates to the image and put it in an xarray DataArray
# The coordinates are latitude and longitude, where latitude spans from 90 degrees north to 56 degrees south,
# and longitude spans from 180 west to 180 east
# The resolution is inferred from the image shape
dataset = [
    xr.DataArray(
        im,
        coords=[np.linspace(90, -56, im.shape[0]), np.linspace(-180, 180, im.shape[1])],
        dims=["latitude", "longitude"],
    )
    for im in images
]

dataset = xr.Dataset(
    data_vars=dict(
        zip(
            [
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
                "Water",
            ],
            dataset,
        )
    )
)


# create a list of land use types
land_use_list = [
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
    "Water",
]

# create a for loop to loop through the land use list, and assign th e values to mos_df as new column
for land_use in land_use_list:
    land_use_da = dataset[land_use].sel(
        latitude=latitude_da, longitude=longitude_da, method="nearest"
    )
    cdc_df[land_use] = land_use_da.values

# close the dataset
dataset.close()

# save the dataframe as a csv file
cdc_df.to_csv(
    "/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/cdc_human_1999_to_2023/WNV_human_and_non_human_yearly_climate.csv",
    index=False,
)

