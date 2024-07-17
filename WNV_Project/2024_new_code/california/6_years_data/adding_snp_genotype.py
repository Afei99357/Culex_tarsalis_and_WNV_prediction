import pandas as pd
from geopy.distance import geodesic

# read the california dataset
california_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/WNV/cali_week_wnnd_multi_years_all_features_mosquitoes_bird.csv",
    header=0,
    index_col=0,
    sep=",",
)

## rename County_Seat_Latitude to latitude and County_Seat_Longitude to longitude
california_df = california_df.rename(
    columns={"County_Seat_Latitude": "Latitude", "County_Seat_Longitude": "Longitude"}
)


## read snp info
snp_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/WNV/pop_culex_result/gene_match_snp_matrix.csv",
    header=0,
    index_col=0
)

## rename GPS.Lat to latitude and GPS.Lon to longitude
snp_df = snp_df.rename(
    columns={"GPS.Lat": "Latitude", "GPS.Lon": "Longitude"}
)

## based on the coordinates in california_df, find the closet geographic distance point in snp_df, and add the vcfID to california_df

# Function to find the closest vcfID
def find_closest_vcfID(lat, lon, df):
    min_distance = float('inf')
    closest_vcfID = None

    for _, row in df.iterrows():
        distance = geodesic((lat, lon), (row['Latitude'], row['Longitude'])).kilometers
        if distance < min_distance:
            min_distance = distance
            closest_vcfID = row['vcfID']

    return closest_vcfID

### add a new column to california_df called vcfID
california_df["vcfID"] = ""

# Update vcfID in df1 with the closest vcfID from df2
for index, row in california_df.iterrows():
    ## print the index
    print(index)
    california_df.at[index, 'vcfID'] = find_closest_vcfID(row['Latitude'], row['Longitude'], snp_df)

## drop the columns that are not needed
snp_df = snp_df.drop(columns=['locID', 'popID', 'State', 'City', 'region', "Latitude", "Longitude"])

## merge california_df and snp_df based on vcfID
california_df = pd.merge(
    california_df,
    snp_df,
    how="left",
    on=["vcfID"],
)

## drop the columns that are not needed
california_df = california_df.drop(columns=['vcfID'])

## save the california_df as a csv file
california_df.to_csv(
    "/Users/ericliao/Desktop/WNV_project_files/WNV/california/data/cali_week_wnnd_with_snp_genotype.csv",
    header=True,
    index=False,
    sep=",",
)