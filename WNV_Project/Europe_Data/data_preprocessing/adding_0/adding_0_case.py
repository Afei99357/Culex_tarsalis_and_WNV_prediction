import pandas as pd
import geopandas as gpd

# read disease data file
euro_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/WNV/Europe_data/europe_data_with_coordinates.csv",
    index_col=0,
)

# Create a DataFrame with all combinations of RegionCode and Time
# get the unique RegionCode
unique_region_code = euro_df["RegionCode"].unique()
# get the unique Time
unique_time = euro_df["Time"].unique()

all_combinations = pd.MultiIndex.from_product([unique_region_code, unique_time], names=['RegionCode', 'Time']).to_frame(index=False)

# Merge with the original dataset
merged = all_combinations.merge(euro_df, on=['RegionCode', 'Time'], how='left')

# Fill missing disease counts with 0
merged['NumValue'].fillna(0, inplace=True)

## choose the columns that we want to keep, RegionCode, Time and NumValue
merged = merged[['RegionCode', 'Time', 'NumValue']]


## adding coordinates
nuts_df = gpd.read_file("/Users/ericliao/Desktop/WNV_project_files/WNV/Europe_data/NUTS_LB_2013_4326.geojson")

# Assuming that the geometry is polygonal, get the centroid
# for each row in eur_df, get the RegionCode, and find the centroid in nuts_df
# and get the latitude and longitude
merged["Latitude"] = ""
merged["Longitude"] = ""
for index, row in merged.iterrows():
    # get the region code
    region_code = row["RegionCode"]
    # find the row in nuts_df that has the same region code
    nuts_row = nuts_df.loc[nuts_df["id"] == region_code]
    # if there is a match, fill in the latitude and longitude
    if nuts_row.shape[0] > 0:
        merged.loc[index, "Latitude"] = nuts_row.iloc[0]["geometry"].centroid.y
        merged.loc[index, "Longitude"] = nuts_row.iloc[0]["geometry"].centroid.x
    else:
        print("No match for RegionCode:", region_code)

## transform the latitude and longitude from EPSG:3035 to EPSG:4326

## reindex the dataframe
eur_df = merged.reset_index(drop=True)

## save the data
eur_df.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/Europe_data/adding_0_case/europe_data_with_coordinates_0_case.csv")

