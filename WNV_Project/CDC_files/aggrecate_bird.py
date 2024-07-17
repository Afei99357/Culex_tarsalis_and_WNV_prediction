import pandas as pd
import os
import numpy as np
import glob

# for loop all the txt files
txt_list = glob.glob("/Users/ericliao/Desktop/WNV_project_files/birds_data/ebirds/ebirds_species/*.txt")

# for loop to iterate each file
for txt_dir in txt_list:

    # get the file name
    file_name = os.path.basename(txt_dir).split(".")[0].split(".")[0]

    # create a column list
    column_list = ["COMMON NAME", "OBSERVATION COUNT", "STATE", "COUNTY", "OBSERVATION DATE",
                   "APPROVED"]

    df_bird = pd.read_csv(txt_dir, sep="\t", usecols=column_list)

    # get the bird name
    bird_name = np.unique(df_bird['COMMON NAME'].values.tolist())[0]

    # only read the data where observation count is X
    df_present = df_bird[df_bird["OBSERVATION COUNT"] == "X"]

    # for the bird presented which has no counts
    df_present["OBSERVATION DATE"] = pd.to_datetime(df_present["OBSERVATION DATE"])
    df_present["MONTH"] = df_present["OBSERVATION DATE"].dt.month
    df_present["YEAR"] = df_present["OBSERVATION DATE"].dt.year

    ## based on the year and county and state, remove the duplicates
    df_present = df_present.drop_duplicates(subset=["YEAR", 'MONTH', "COUNTY"])
    # drop date and Approved columns
    df_present = df_present.drop(columns=["OBSERVATION DATE", "APPROVED"])

    # replace X with nan for OBSERVATION COUNT
    df_present["OBSERVATION COUNT"] = df_present["OBSERVATION COUNT"].replace("X", None)


    ############# for bird has actualy counts #####################
    # # drop the rows if the Observation Count is X
    df_bird_count = df_bird[(df_bird["OBSERVATION COUNT"] != "X") & (df_bird["APPROVED"] == 1)]

    # # specify the column data type for each column
    column_dtype = {"COMMON NAME": str, "OBSERVATION COUNT": float, "OBSERVATION DATE": str, "APPROVED": int}

    # change the data type for each column in df
    df_bird_count = df_bird_count.astype(column_dtype)

    # # add up the observation count for each month absed on the Obervation date
    df_bird_count["OBSERVATION DATE"] = pd.to_datetime(df_bird_count["OBSERVATION DATE"])
    df_bird_count["MONTH"] = df_bird_count["OBSERVATION DATE"].dt.month
    df_bird_count["YEAR"] = df_bird_count["OBSERVATION DATE"].dt.year

    # drop date
    df_bird_count = df_bird_count.drop(columns=["OBSERVATION DATE"])

    # # group by the month and year and sum up the observation count
    df_bird_count = df_bird_count.groupby(["YEAR", "MONTH", "COUNTY", "STATE"]).sum().reset_index()

    # add common name column
    df_bird_count["COMMON NAME"] = bird_name

    # # add one column present_flag for both df_bird_count and df_present, 0 for df_bird_count and 1 for df_present
    df_bird_count["PRESENT_FLAG"] = 0
    df_present["PRESENT_FLAG"] = 1

    # # concat df_bird_count and df_present
    df_bird_total = pd.concat([df_bird_count, df_present], axis=0)

    # # order the columns
    df_bird_total = df_bird_total[["COMMON NAME", "YEAR", "MONTH", "STATE", "COUNTY", "OBSERVATION COUNT", "PRESENT_FLAG"]]

    # # # output as csv
    # df_bird_total.to_csv(f"/Users/ericliao/Desktop/WNV_project_files/birds_data/ebirds/ebirds_species/"
    #                      f"aggregate_files/{file_name}.csv")

    # # output as parquet
    df_bird_total.to_parquet(f"/Users/ericliao/Desktop/WNV_project_files/birds_data/ebirds/ebirds_species/"
                                f"aggregate_files/{file_name}.parquet", engine="pyarrow")


