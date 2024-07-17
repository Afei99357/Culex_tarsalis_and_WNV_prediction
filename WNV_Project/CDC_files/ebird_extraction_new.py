import pandas as pd
import glob
import numpy as np

# # read the csv file
disease_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/"
                         "cdc_sum_organism_climate_binary.csv", index_col=False)

# convert State and County to lower case and strip it
disease_df["County"] = disease_df["County"].str.lower()
disease_df["State"] = disease_df["State"].str.lower()
disease_df["County"] = disease_df["County"].str.strip()
disease_df["State"] = disease_df["State"].str.strip()

# loop through all the parquet files
parquet_list = glob.glob("/Users/ericliao/Desktop/WNV_project_files/birds_data/ebirds/"
                         "ebirds_species/aggregate_files/*.parquet")



for file in parquet_list:
    # read the parquet file
    df_bird = pd.read_parquet(file)

    # convert State and County to lower case and strip it
    df_bird["COUNTY"] = df_bird["COUNTY"].str.lower()
    df_bird["STATE"] = df_bird["STATE"].str.lower()
    df_bird["COUNTY"] = df_bird["COUNTY"].str.strip()
    df_bird["STATE"] = df_bird["STATE"].str.strip()

    # get the bird name
    bird_name = np.unique(df_bird['COMMON NAME'].values.tolist())[0]

    # only read the data where observation count is not nan
    df_bird_count = df_bird[df_bird["OBSERVATION COUNT"].notna()]

    # merge the count
    merged_df = pd.merge(disease_df, df_bird_count, how='left', left_on=["Year", "Month", "State", "County"],
                         right_on=["YEAR", "MONTH", "STATE", "COUNTY"])

    # drop the columns that are not needed
    disease_df = merged_df.drop(columns=["COMMON NAME", "YEAR", "MONTH", "STATE", "COUNTY", "PRESENT_FLAG"])

    # rename OBSERVATION COUNT to bird name
    disease_df = disease_df.rename(columns={"OBSERVATION COUNT": bird_name})

    # get the present flag is 1
    df_present = df_bird[df_bird["PRESENT_FLAG"] == 1]

    # check for present and, if not present just add 0 for nan
    # print notification
    print("Beginning check nan values ", bird_name)

    # join disease_df_new and df_present, based on Year, Month, State and County
    df_join = pd.merge(disease_df, df_present, how='left', left_on=["Year", "Month", "State", "County"],
                       right_on=["YEAR", "MONTH", "STATE", "COUNTY"])

    # for the rows in df_join where the bird name is nan, if the present flag is nan, then replace nan in bird name with 0
    df_join[bird_name].loc[pd.isnull(df_join[bird_name]) & pd.isnull(df_join["PRESENT_FLAG"])] = 0

    # drop the columns that are not needed
    disease_df = df_join.drop(columns=["COMMON NAME", "YEAR", "MONTH", "STATE", "COUNTY", "OBSERVATION COUNT",
                                           "PRESENT_FLAG"])

    # release memory
    del df_bird
    del df_bird_count
    del df_present

    # print notification
    print("Finished adding bird: ", bird_name)



# save the file
disease_df.to_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/"
                  "cdc_sum_organism_all_binary.csv")
