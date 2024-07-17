import pandas as pd

# load the data
df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/"
                 "non_human_data_2002_2017_monthly_aggregated_by_county.csv", index_col=False)

## for mosquitoes
# # drop column bird and horse
df_mosquitoes = df.drop(["Horse", "Mosquitoes"], axis=1)

# # drop nan values
df_mosquitoes = df_mosquitoes.dropna()

# # group by data based on Year, Month, County
df_mosquitoes = df_mosquitoes.groupby(["Year", "Month", "County"]).sum().reset_index()

# # add county seat coordinates
df_county_seat = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/"
                             "Data/human_neuroinvasive_wnv_ebirds.csv", index_col=0)
# # state in Illinois
df_county_seat = df_county_seat[df_county_seat["State"] == "Illinois"]

# # only get the county seat coordinates and county name
df_county_seat = df_county_seat[["County", "FIPS", "County_Seat_Longitude", "County_Seat_Latitude",
                                 "Poverty_Estimate_All_Ages", "Population", "Land_Area_2010", "Avian Phylodiversity"]]

# # get unique pairs of county and county seat coordinates
df_county_seat = df_county_seat.drop_duplicates()

# # lower case the values in County and strip the white space
df_mosquitoes["County"] = df_mosquitoes["County"].str.lower()
df_mosquitoes["County"] = df_mosquitoes["County"].str.strip()
df_county_seat["County"] = df_county_seat["County"].str.lower()
df_county_seat["County"] = df_county_seat["County"].str.strip()

# # merge the data
df_mosquitoes = pd.merge(df_mosquitoes, df_county_seat, how="left", on="County")

## output the data
df_mosquitoes.to_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/county_level_data_2018_2022/"
          "bird_illnois_aggregate_county.csv")