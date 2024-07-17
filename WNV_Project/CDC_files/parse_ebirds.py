import pandas as pd
import os
import glob

# load birds files
# create file list for all the .txt files
# file_list = glob.glob("/Users/ericliao/Desktop/WNV_project_files/birds_data/ebirds/ebirds_species/*.txt")

# create a column list
column_list = ["COMMON NAME", "OBSERVATION COUNT", "STATE", "COUNTY", "OBSERVATION DATE", "APPROVED"]

# read the SECOND file
# df = pd.read_csv(file_list[1], sep="\t", usecols=column_list)

df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/birds_data/ebirds/ebirds_species/"
                 "ebd_US_northernCardinal_199901_202301_relJan-2023.txt", sep="\t", usecols=column_list)


# #  get the present birds data
df_present = df[df["OBSERVATION COUNT"] == "X"]

# # # only California
# df_present = df_present[df_present["STATE"] == "California"]

# interpret date
# # add up the observation count for each month absed on the Obervation date
df_present["OBSERVATION DATE"] = pd.to_datetime(df_present["OBSERVATION DATE"])
df_present["MONTH"] = df_present["OBSERVATION DATE"].dt.month
df_present["YEAR"] = df_present["OBSERVATION DATE"].dt.year

## based on the year and county and state, remove the duplicates
df_present = df_present.drop_duplicates(subset=["YEAR", 'MONTH', "STATE", "COUNTY"])

# # drop the rows if the Observation Count is X
df = df[(df["OBSERVATION COUNT"] != "X") & (df["APPROVED"] == 1)]

# # specify the column data type for each column
column_dtype = {"COMMON NAME": str, "OBSERVATION COUNT": int, "STATE": str, "COUNTY": str, "OBSERVATION DATE": str, "APPROVED": str}

# change the data type for each column in df
df = df.astype(column_dtype)

# # add up the observation count for each month absed on the Obervation date
df["OBSERVATION DATE"] = pd.to_datetime(df["OBSERVATION DATE"])
df["MONTH"] = df["OBSERVATION DATE"].dt.month
df["YEAR"] = df["OBSERVATION DATE"].dt.year

# drop month
df = df.drop(columns=["OBSERVATION DATE"])

# # group by the month and year and sum up the observation count
df = df.groupby(["YEAR", "MONTH", "STATE", "COUNTY"]).sum().reset_index()

# READ the disease dataset
disease_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/robinA.csv",
                         index_col=0)

# convert State and County to lower case
disease_df["State"] = disease_df["State"].str.lower()
disease_df["County"] = disease_df["County"].str.lower()
df["STATE"] = df["STATE"].str.lower()
df["COUNTY"] = df["COUNTY"].str.lower()

# remove the space in the beginning and end of the string
disease_df["State"] = disease_df["State"].str.strip()
disease_df["County"] = disease_df["County"].str.strip()
df["STATE"] = df["STATE"].str.strip()
df["COUNTY"] = df["COUNTY"].str.strip()


## merge the two dataframes based on the year and county and state, month
disease_df = pd.merge(disease_df, df, how="left", left_on=["Year", "Month", "State", "County"], right_on=["YEAR", "MONTH", "STATE", "COUNTY"])

# # drop the columns that are not needed
disease_df = disease_df.drop(columns=["YEAR", 'MONTH', "STATE", "COUNTY"])

## for loop each row in disease_df, if the Observation count is null, if this row is not match any rows in present_df based on the year and county and state, fill in 0
for i in range(len(disease_df)):
    if pd.isnull(disease_df.iloc[i]["OBSERVATION COUNT"]):
        if df_present[(df_present["YEAR"] == disease_df.iloc[i]["Year"]) &
                      (df_present["MONTH"] == disease_df.iloc[i]["Month"]) &
                      (df_present["STATE"] == disease_df.iloc[i]["State"]) &
                      (df_present["COUNTY"] == disease_df.iloc[i]["County"])].empty:
            disease_df.loc[i, "OBSERVATION COUNT"] = 0

# rename Observation Count to Green Jay
disease_df = disease_df.rename(columns={"OBSERVATION COUNT": "Northern Cardinal"})

disease_df.to_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/combine_cdc_all_birds.csv", index=True)