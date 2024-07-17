import pandas as pd

## import human data
human_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/cdc_human_1999_to_2023/"
                "West_Nile_virus_human_and_non-human_activity_by_county_1999_to_2023_CDC.csv", index_col=False, header=0)

## select the Year is 2023, column Reported human cases fill with the values from column Total human disease casees
human_df.loc[human_df["Year"] == 2023, "Reported human cases"] = human_df.loc[human_df["Year"] == 2023, "Total human disease cases"]

## select the Year is 2023, column Identified by Blood Donor Screening fill with the values from column Presumptive viremic blood donors
human_df.loc[human_df["Year"] == 2023, "Identified by Blood Donor Screening"] = human_df.loc[human_df["Year"] == 2023, "Presumptive viremic blood donors"]

## remove columns Total human disease cases and Presumptive viremic blood donors
human_df = human_df.drop(columns=["Total human disease cases", "Presumptive viremic blood donors"])

## save the dataframe to a csv file
human_df.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/cdc_human_1999_to_2023/"
                "West_Nile_virus_human_and_non-human_activity_by_county_1999_to_2023_CDC_correct.csv", index=False)



