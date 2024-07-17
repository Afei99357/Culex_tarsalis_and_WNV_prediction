import pandas as pd
import os
import glob

# create a column list
column_list = ["COMMON NAME", "OBSERVATION COUNT", "STATE", "COUNTY", "LATITUDE", "LONGITUDE", "OBSERVATION DATE", "APPROVED"]

# read the SECOND file
# df = pd.read_csv(file_list[1], sep="\t", usecols=column_list)

# for loop all the txt files
txt_list = glob.glob("/Users/ericliao/Desktop/WNV_project_files/birds_data/ebirds/ebirds_species/*.txt")

# for loop
for txt in txt_list:
    # read the txt file in pandas
    df = pd.read_csv(txt, sep="\t", usecols=column_list)

    # get the bird name
    bird_name = os.path.basename(txt).split(".")[0].split("_")[2]

    # df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/birds_data/ebirds/ebirds_species/"
    #                  "ebd_US_grycat_199901_202301_relFeb-2023.txt", sep="\t", usecols=column_list)

    # only get illinois data
    df = df[df['STATE'] == 'Illinois']

    # remove state column from df
    df = df.drop(columns=["STATE"])

    # #  get the present birds data
    df_present = df[df["OBSERVATION COUNT"] == "X"]

    # interpret date
    # # add up the observation count for each month absed on the Obervation date
    df_present["OBSERVATION DATE"] = pd.to_datetime(df_present["OBSERVATION DATE"])
    df_present["MONTH"] = df_present["OBSERVATION DATE"].dt.month
    df_present["YEAR"] = df_present["OBSERVATION DATE"].dt.year

    ## based on the year and county and state, remove the duplicates
    df_present = df_present.drop_duplicates(subset=["YEAR", 'MONTH', "COUNTY"])

    # # drop the rows if the Observation Count is X
    df = df[(df["OBSERVATION COUNT"] != "X") & (df["APPROVED"] == 1)]

    # # specify the column data type for each column
    column_dtype = {"COMMON NAME": str, "OBSERVATION COUNT": int, "LATITUDE": float, "LONGITUDE": float,
                    "OBSERVATION DATE": str, "APPROVED": str}

    # change the data type for each column in df
    df = df.astype(column_dtype)

    # # add up the observation count for each month absed on the Obervation date
    df["OBSERVATION DATE"] = pd.to_datetime(df["OBSERVATION DATE"])
    df["MONTH"] = df["OBSERVATION DATE"].dt.month
    df["YEAR"] = df["OBSERVATION DATE"].dt.year

    # drop month
    df = df.drop(columns=["OBSERVATION DATE"])

    # # group by the month and year and sum up the observation count
    df = df.groupby(["YEAR", "MONTH", "COUNTY"]).sum().reset_index()

    ## add a for loop to go through three csv files
    # # create a list of csv files
    csv_list = glob.glob("/Users/ericliao/Desktop/WNV_project_files/illinois_data/county_level_data_2018_2022/climate/*.csv")

    # # for loop each csv file
    for csv_file in csv_list:
        # # read the csv file
        disease_df = pd.read_csv(csv_file, index_col=False)

        # # get file name
        file_name = os.path.basename(csv_file).split(".")[0]

        # convert State and County to lower case
        disease_df["County"] = disease_df["County"].str.lower()
        df["COUNTY"] = df["COUNTY"].str.lower()

        # remove the space in the beginning and end of the string
        disease_df["County"] = disease_df["County"].str.strip()
        df["COUNTY"] = df["COUNTY"].str.strip()

        ## merge the two dataframes based on the year and county and state, month
        disease_df = pd.merge(disease_df, df, how="left", left_on=["Year", "Month", "County"], right_on=["YEAR", "MONTH", "COUNTY"])

        # # drop the columns that are not needed
        disease_df = disease_df.drop(columns=['MONTH',  "COUNTY"])

        ## for loop each row in disease_df, if the Observation count is null, if this row is not match any rows in present_df based on the year and county and state, fill in 0
        for i in range(len(disease_df)):
            if pd.isnull(disease_df.iloc[i]["OBSERVATION COUNT"]):
                if df_present[(df_present["YEAR"] == disease_df.iloc[i]["Year"]) &
                              (df_present["MONTH"] == disease_df.iloc[i]["Month"]) &
                              (df_present["COUNTY"] == disease_df.iloc[i]["County"])].empty:
                    disease_df.loc[i, "OBSERVATION COUNT"] = 0

        # rename Observation Count to Green Jay
        disease_df = disease_df.rename(columns={"OBSERVATION COUNT": f"{bird_name}"})

        # remove the useless column
        disease_df = disease_df.drop(columns=["YEAR", "LATITUDE", "LONGITUDE"])

        disease_df.to_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/county_level_data_2018_2022/climate/"
                          f"{file_name}.csv", index=False)

        ## PRINT OUT NOTIFICATION
        print(f"Finished processing {file_name}.csv")

        # clean up the memory
        del disease_df
    # PRTINT OUT NOTIFICATION
    print(f"Finished processing {bird_name}.txt")
    del df