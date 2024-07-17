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
                 "ebd_US_norcar_199901_202301_relJan-2023.txt", sep="\t", usecols=column_list)

# #  get the present birds data
df_present = df[df["OBSERVATION COUNT"] == "X"]

# interpret date
# # add up the observation count for each month absed on the Obervation date
df_present["OBSERVATION DATE"] = pd.to_datetime(df_present["OBSERVATION DATE"])
df_present["MONTH"] = df_present["OBSERVATION DATE"].dt.month
df_present["YEAR"] = df_present["OBSERVATION DATE"].dt.year

## based on the year and county and state, remove the duplicates
df_present = df_present.drop_duplicates(subset=["YEAR", "STATE", "COUNTY"])

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
df = df.drop(columns=["MONTH"])

# # group by the month and year and sum up the observation count
df = df.groupby(["YEAR", "STATE", "COUNTY"]).sum().reset_index()

# READ the disease dataset
disease_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/ebirds_robin.csv",
                         index_col=0)

## merge the two dataframes
disease_df = pd.merge(disease_df, df, how="left", left_on=["Year", "State", "County"], right_on=["YEAR", "STATE", "COUNTY"])

# # drop the columns that are not needed
disease_df = disease_df.drop(columns=["YEAR", "STATE", "COUNTY"])

## for loop each row in disease_df, if the Observation count is null, if this row is not match any rows in present_df based on the year and county and state, fill in 0
for i in range(len(disease_df)):
    if pd.isnull(disease_df.iloc[i]["OBSERVATION COUNT"]):
        if df_present[(df_present["YEAR"] == disease_df.iloc[i]["Year"]) &
                      (df_present["STATE"] == disease_df.iloc[i]["State"]) &
                      (df_present["COUNTY"] == disease_df.iloc[i]["County"])].empty:
            disease_df.loc[i, "OBSERVATION COUNT"] = 0

# rename Observation Count to Green Jay
disease_df = disease_df.rename(columns={"OBSERVATION COUNT": "Northern Cardinal"})

disease_df.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/human_neuroinvasive_wnv_ebirds.csv", index=True)