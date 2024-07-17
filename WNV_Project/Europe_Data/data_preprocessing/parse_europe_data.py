import pandas as pd
import geopandas as gpd

# load the data
eur_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/Europe_data/"
                     "ECDC_surveillance_data_West_Nile_virus_infection.csv")

nuts_df = gpd.read_file("/Users/ericliao/Desktop/WNV_project_files/WNV/Europe_data/NUTS_LB_2013_4326.geojson")

# Assuming that the geometry is polygonal, get the centroid
# for each row in eur_df, get the RegionCode, and find the centroid in nuts_df
# and get the latitude and longitude
eur_df["Latitude"] = ""
eur_df["Longitude"] = ""
for index, row in eur_df.iterrows():
    # get the region code
    region_code = row["RegionCode"]
    # find the row in nuts_df that has the same region code
    nuts_row = nuts_df.loc[nuts_df["id"] == region_code]
    # if there is a match, fill in the latitude and longitude
    if nuts_row.shape[0] > 0:
        eur_df.loc[index, "Latitude"] = nuts_row.iloc[0]["geometry"].centroid.y
        eur_df.loc[index, "Longitude"] = nuts_row.iloc[0]["geometry"].centroid.x
    else:
        print("No match for RegionCode:", region_code)

## transform the latitude and longitude from EPSG:3035 to EPSG:4326

## reindex the dataframe
eur_df = eur_df.reset_index(drop=True)

# save the data
eur_df.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/Europe_data/europe_data_with_coordinates.csv")




