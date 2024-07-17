import pandas as pd
import xarray as xr
import numpy as np


### create data array to store the california weekly data ###########
# read data
df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/"
                 "human_neuroinvasive_wnv_rate_log_population_correct_poverty_mean_green_index.csv")

# get only california
df = df[df['State'] == "California"]
## unique county
unique_county = df['County'].unique()

## use xarray to create a dataarray with counties, years, weeks
## create a dataarray with counties, years, weeks
# create a list of counties
county_list = df['County'].unique().tolist()

# create a list of years from 1999 to 2022
year_list = list(range(1999, 2022))

# create a list of weeks from 1 to 52
week_list = list(range(1, 53))

# create a dictionary of attributes
species_list = ["Humans", 'Horses', 'Dead Birds', 'Mosquito Pools', 'Sentinel Chicken', "Squirrels"]

# create a data array with counties, years, weeks
ca_data_array = xr.DataArray(data=np.zeros((len(county_list), len(year_list), len(week_list), len(species_list))),
                             dims=["county", "year", "week", "species"],
                             coords={"county": county_list, "year": year_list, "week": week_list, "species": species_list})

# read a table in the pdf file

