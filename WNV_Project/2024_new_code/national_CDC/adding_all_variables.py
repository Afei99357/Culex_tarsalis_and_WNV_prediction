import pandas as pd
import xarray as xr
import cv2
import numpy as np


######### adding phylogenetic diversity index to the cdc dataset  ############
# load data
cdc_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/monthly/combine_cdc_all_no_climate.csv",
    index_col=0,
)

## create a column to combine the year, month as two digits, and date 1st of every month
cdc_df["Date"] = (
    cdc_df["Year"].astype(str) + "-" + cdc_df["Month"].astype(str).str.zfill(2) + "-01"
)

## ## Based on Date and FIPS, drop the duplicated rows
cdc_df = cdc_df.drop_duplicates(subset=["Date", "FIPS"])

## sort the dataframe based on the Date
cdc_df = cdc_df.sort_values(by="Date")

## reindex the dataframe
cdc_df = cdc_df.reset_index(drop=True)

print("start adding phylodiversity")
df_phylodiversity = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/WNV/bird/avi_phylodiv_wnv_041822.csv",
    index_col=0,
)

# based on the FIPS in df and STCO_FIPS in df_phylodiversity, merge the phylodiversity index to the cdc dataset
cdc_df = cdc_df.merge(
    df_phylodiversity, left_on="FIPS", right_on="STCO_FIPS", how="left"
)

# drop the STCO_FIPS column
cdc_df = cdc_df.drop(columns="STCO_FIPS")

################ adding climate variables to the cdc dataset   ####################
print("start adding climate variables")

ds = xr.open_dataset(
    "/Users/ericliao/Downloads/new_land_monthly_data_2024_20_15.nc"
)

## sort the ds based on time
ds = ds.sortby("time")

print(ds)
## access ds u10 variable where time is all time and the area is belonged to U.S. Contiguous 48 United States
## (http://www.cohp.org/extremes/extreme_points.html)
## The latitude range is between 25.1 N and 49.4 N and the longitude range is between 66.9 W and 124.7 W
## access all variables from ds where time is all time and the area is belonged to U.S. Contiguous 48 United States

# create a data array based on the latitude from human_df
latitude_da = xr.DataArray(cdc_df["County_Seat_Latitude"].values, dims="county")
# create a data array based on the longitude from human_df
longitude_da = xr.DataArray(cdc_df["County_Seat_Longitude"].values, dims="county")
# create a data array based on the date from nonhuman_df
time_da = xr.DataArray(cdc_df["Date"].values.astype("datetime64[D]"), dims="county")

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

new_variable_name_list = [
    "u10_1m_shift",
    "v10_1m_shift",
    "t2m_1m_shift",
    "lai_hv_1m_shift",
    "lai_lv_1m_shift",
    "src_1m_shift",
    "sf_1m_shift",
    "ssr_1m_shift",
    "sro_1m_shift",
    "e_1m_shift",
    "tp_1m_shift",
    "swvl1_1m_shift"

]

# select each variable separately
for variable, new_variable in zip(variable_list, new_variable_name_list):
    print("start ", variable)
    alltime_all_var_values = ds[variable].sel(
        latitude=latitude_da, longitude=longitude_da, method="nearest"
    )

    alltime_all_var_values_shift_1 = alltime_all_var_values.shift(time=1).sel(
        time=time_da, method="nearest"
    )

    cdc_df[new_variable] = alltime_all_var_values_shift_1.values
    alltime_all_var_values.close()
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
    "/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/monthly/combine_cdc_all_environmental_variable_all_2024.csv",
    index=False,
)

