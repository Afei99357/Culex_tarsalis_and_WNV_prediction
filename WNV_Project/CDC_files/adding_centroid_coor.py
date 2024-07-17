import pandas as pd

# load the data
centroid_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/US_county_info/"
                          "us_county_centroid_coordinates_with_fips.csv", index_col=0)

# separate the coordinates to latitude and longitude by comma and remove space
centroid_df["Latitude"] = centroid_df["Geo Point"].apply(lambda x: x.split(",")[0].strip())
centroid_df["Longitude"] = centroid_df["Geo Point"].apply(lambda x: x.split(",")[1].strip())

# load the dataset has missing coordinates of county seat
cdc_sum_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/"
                         "cdc_sum_organism_with_coordinates.csv", index_col=0)

# choose the rows that coordinates are nan
cdc_sum_df_nan = cdc_sum_df.loc[pd.isna(cdc_sum_df["Latitude"]) | pd.isna(cdc_sum_df["Longitude"])]
cdc_sum_df_not_nan = cdc_sum_df.loc[~(pd.isna(cdc_sum_df["Latitude"]) | pd.isna(cdc_sum_df["Longitude"]))]

# if the latitude and longitude are nan, looking for the centroid dataset to see if there is a FIPS match, if does, fill
# in the latitude and longitude
for index, row in cdc_sum_df_nan.iterrows():
    # get the FIPS code
    fips = row["FIPS"]
    # find the row in the centroid dataset that has the same FIPS code
    centroid_row = centroid_df.loc[centroid_df["FIPS"] == fips]
    # if there is a match, fill in the latitude and longitude
    if centroid_row.shape[0] > 0:
        cdc_sum_df_nan.loc[index, "Latitude"] = centroid_row.iloc[0]["Latitude"]
        cdc_sum_df_nan.loc[index, "Longitude"] = centroid_row.iloc[0]["Longitude"]
    else:
        print("No match for FIPS:", fips)

# combine the two dataframes
cdc_sum_df = pd.concat([cdc_sum_df_nan, cdc_sum_df_not_nan])
# save the data
cdc_sum_df.to_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/cdc_sum_organism_fill_centroid.csv")
