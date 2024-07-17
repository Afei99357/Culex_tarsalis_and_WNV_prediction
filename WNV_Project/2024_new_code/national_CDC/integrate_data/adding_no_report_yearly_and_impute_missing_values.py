import pandas as pd

## import both human and nonhuman data
human_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/cdc_human_1999_to_2023/"
                "WNV_human_and_non-human_annual_by_county_1999_to_2023.csv", index_col=False, header=0)

## only keep Year fromm 2003 to 2023
human_df = human_df[(human_df["Year"] >= 2004) & (human_df["Year"] <= 2023)]

## get a subset of "FIPS", "State", "County", "Latitude", "Longitude"
county_info = human_df[["FIPS", "State", "County", "Latitude", "Longitude"]]

## drop the duplicated rows based on FIPS
county_info = county_info.drop_duplicates(subset=["FIPS"])

## drop the rows where FIPS is nan
human_df = human_df.dropna(subset=["FIPS"])

## drop the rows where State is nan
human_df = human_df.dropna(subset=["State"])

## get unique FIPS
unique_fips = human_df["FIPS"].unique()

## create an adataframe, shape is size of unique_fips * number of year from 2004 to 2023,
# for each FIPS in unique FIPS, adding a row for each year from 2004 to 2023
numeber_of_year = 2023 - 2004 + 1

df = pd.DataFrame(index=range(len(unique_fips)*numeber_of_year), columns=["FIPS", "Year"])
df["FIPS"] = [fips for fips in unique_fips for _ in range(numeber_of_year)]
df["Year"] = [year for _ in range(len(unique_fips)) for year in range(2004, 2024)]

## adding State, County, Latitude, Longitude to the dataframe
df = df.merge(county_info, on="FIPS", how="left")

## merge the human and non-human data based on FIPS and Year
df = df.merge(human_df, on=["FIPS", "Year"], how="left")

## rename the columns
df = df.rename(columns={"State_x": "State", "County_x": "County", "Latitude_x": "Latitude", "Longitude_x": "Longitude"})

## drop the duplicated columns
df = df.drop(columns=["State_y", "County_y", "Latitude_y", "Longitude_y"])

## reorder the columns
df = df[["FIPS", "Year", "State", "County", "Latitude", "Longitude", "Activity", 'Reported human cases',
         'Neuroinvasive disease cases', 'Identified by Blood Donor Screening', 'Total_Bird_WNV_Count',
         "WNV_Corvid_Count", "WNV_NonCorvid_Count", 'Mos_WNV_Count', 'Horse_WNV_Count']]

## rename the columns
df = df.rename(columns={"Reported human cases": "Reported_human_cases", "Neuroinvasive disease cases": "Neuroinvasive_disease_cases",
                        "Identified by Blood Donor Screening": "Identified_by_Blood_Donor_Screening"})

## for each row in the df, if Activity is nan, fill 0 with 'Reported human cases', 'Neuroinvasive disease cases',
# 'Identified by Blood Donor Screening', 'Total_Bird_WNV_Count',
# "WNV_Corvid_Count", "WNV_NonCorvid_Count", 'Mos_WNV_Count', 'Horse_WNV_Count'
df.loc[df["Activity"].isna(), ['Reported_human_cases', 'Neuroinvasive_disease_cases', 'Identified_by_Blood_Donor_Screening',
                                'Total_Bird_WNV_Count', "WNV_Corvid_Count", "WNV_NonCorvid_Count", 'Mos_WNV_Count', 'Horse_WNV_Count']] = 0

## for each row in the df, if Activity is "Non-human activity" and "Neuroinvasive disease cases" is nan, fill 0 with 'Neuroinvasive disease cases'
df.loc[(df["Activity"] == "Non-human activity") & (df["Neuroinvasive_disease_cases"].isna()), "Neuroinvasive_disease_cases"] = 0

## adding an column for "average_human_case_over_20_years"
df["average_human_case_over_20_years"] = df.groupby("FIPS")["Reported_human_cases"].transform("mean")

## for each row, if the "Reported_human_cases" is nan, fill "Reported_human_cases" with "average_human_case_over_20_years"
df.loc[df["Reported_human_cases"].isna(), "Reported_human_cases"] = df.loc[df["Reported_human_cases"].isna(), "average_human_case_over_20_years"]

## output the dataframe to a csv file
df.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/cdc_human_1999_to_2023/"
          "WNV_human_and_non-human_annual_by_county_2004_to_2023_impute_missing.csv", index=False)

print(df.columns.values)





