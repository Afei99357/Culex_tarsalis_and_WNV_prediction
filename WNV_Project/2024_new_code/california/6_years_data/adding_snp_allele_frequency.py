import pandas as pd
from geopy.distance import geodesic

# read the california dataset
california_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/WNV/california/data/cali_week_wnnd_impute_monthly_mean_value.csv",
    header=0,
    index_col=0,
    sep=",",
)

## read snp info
allelic_frequency_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/WNV/california/data/allele_frequency_table_sample_candidate_gene_all.csv",
    header=0,
    index_col=0
)

## only get the candidate snp from RDA
rda_candidate_snp = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/"
                                "Paper_results/compare_landscape_and_outlier_analysis/"
                                "compare_candidate_allel_alt_allele_freq_over_samples/culex_pop_gene_snp_candidates.csv",
                                header=0, index_col=0)

## get the list of alternate_allele
alt_allele_list = rda_candidate_snp["alternate_allele"].tolist()

## filter the allelic_frequency_df to only include the column names that are in the alt_allele_list and popID
allelic_frequency_df = allelic_frequency_df[["popID"] + alt_allele_list]

## read population info
population_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/"
                            "Ctarsalis_sample_w_GPS_climate_average_new_filtered_id_region.csv",
                            header=0,
                            index_col=0)

## subset the population_df to only include the vcfID, locID, region, popID, GPS.Lat and GPS.Lon
population_df = population_df[["vcfID", "popID", "GPS.Lat", "GPS.Lon"]]

## merge population_df and allelic_frequency_df based on popID
snp_df = pd.merge(allelic_frequency_df, population_df, how="left", on=["popID"])

## rearrange the columns, put vcfID, locID, region, popID, GPS.Lat, GPS.Lon first and the oher columns after
snp_df = snp_df[["vcfID", "popID", "GPS.Lat", "GPS.Lon"] + [col for col in snp_df.columns if col not in ["vcfID", "locID", "region", "popID", "GPS.Lat", "GPS.Lon"]]]

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
snp_df = snp_df.drop(columns=['popID', "Latitude", "Longitude"])

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
    "/Users/ericliao/Desktop/WNV_project_files/WNV/california/data/cali_week_impute_monthly_mean_value_with_allele_frequency_RDA.csv",
    header=True,
    index=False,
    sep=",",
)