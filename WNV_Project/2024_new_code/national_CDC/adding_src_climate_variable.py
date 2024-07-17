import xarray as xr
import pandas as pd

cdc_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/monthly/combine_cdc_all_environmental_variable_all_2024.csv")

ds = xr.open_dataset(
    "/Users/ericliao/Downloads/new_land_monthly_data_2024_20_15.nc"
)

## sort the ds based on time
ds = ds.sortby("time")

# create a data array based on the latitude from human_df
latitude_da = xr.DataArray(cdc_df["County_Seat_Latitude"].values, dims="county")
# create a data array based on the longitude from human_df
longitude_da = xr.DataArray(cdc_df["County_Seat_Longitude"].values, dims="county")
# create a data array based on the date from nonhuman_df
time_da = xr.DataArray(cdc_df["Date"].values.astype("datetime64[D]"), dims="county")

print("flag 1")
alltime_all_var_values = ds["src"].sel(latitude=latitude_da, longitude=longitude_da, method="nearest")

print('flag 2')
alltime_all_var_values_shift_1 = alltime_all_var_values.shift(time=1).sel(time=time_da, method="nearest")

cdc_df["src_1m_shift"] = alltime_all_var_values_shift_1.values

## output the cdc_df to a csv file
cdc_df.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/monthly/combine_cdc_all_environmental_variable_all_2024_test.csv")

