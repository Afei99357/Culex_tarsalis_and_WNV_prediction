import pandas as pd

## import human data
human_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/cdc_human_1999_to_2023/"
                "West_Nile_virus_human_and_non-human_activity_by_county_1999_to_2023_CDC_correct.csv", index_col=False, header=0)

## import non-human data
nonhuman_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/monthly/"
                          "combine_cdc_all_environmental_variable_all_2024.csv", index_col=False, header=0)

## keep the columns that are needed from non-human data, WNV_Corvid_Count, WNV_NonCorvid_Count, Total_Bird_WNV_Count,
## Mos_WNV_Count, Horse_WNV_Count, Year, Month, FIPS, State, County, County_Seat_Latitude, County_Seat_Longitude
nonhuman_df = nonhuman_df[["WNV_Corvid_Count", "WNV_NonCorvid_Count", "Total_Bird_WNV_Count", "Mos_WNV_Count",
                           "Horse_WNV_Count", "Year", "FIPS"]]

## aggragate the non-human data based on FIPS, Year
nonhuman_df = nonhuman_df.groupby(["FIPS", "Year"]).sum().reset_index()

## merge the human and non-human data based on FIPS and Year
df = human_df.merge(nonhuman_df, on=["FIPS", "Year"], how="left")

## for rows where Reported human cases are nan, if the column Activity is "Non-human activity", then column Reported human cases fill with 0,
## if the column Activity is "Human activity" or "Human infections and non-human activity", then column Reported human cases fill with nan

## save the dataframe to a csv file
df.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/cdc_human_1999_to_2023/"
          "WNV_human_and_non-human_annual_by_county_1999_to_2023.csv", index=False)


