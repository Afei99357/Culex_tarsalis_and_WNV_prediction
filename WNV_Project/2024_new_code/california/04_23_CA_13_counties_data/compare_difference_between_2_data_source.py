import pandas as pd

# Load the CA dataset into a Pandas DataFrame
data_CA_13 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/CA_13_counties_04_23_no_impute.csv",
                   index_col=False,
                   header=0)

## impute any missing in Human_Disease_Count with 0
data_CA_13["Human_Disease_Count"] = data_CA_13["Human_Disease_Count"].fillna(0)

## only keep columns, County, Year, Month, Human_Disease_Count
data_CA_13 = data_CA_13[["County", "Year", "Month", "Human_Disease_Count"]]

## sort the data by County, Year, Month
data_CA_13 = data_CA_13.sort_values(by=["County", "Year", "Month"])


## load eric data
data_eric_13 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/CA_human_data_2004_to_2023_final_no_impute_0.csv",
                   index_col=False,
                   header=0)

##n choose data only in county: Fresno, Kern, Los Angeles, Merced, Orange, Placer, Riverside, Sacramento, San Bernardino, San Joaquin, Solano, Stanislaus, and Tulare
data_eric_13 = data_eric_13[data_eric_13["County"].isin(x.lower() for x in ["Fresno", "Kern", "Los Angeles", "Merced", "Orange", "Placer", "Riverside",
                                 "Sacramento", "San Bernardino", "San Joaquin", "Solano",
                                 "Stanislaus", "Tulare"])]

## impute any missing in Human_Disease_Count with 0
data_eric_13["Human_Disease_Count"] = data_eric_13["Human_Disease_Count"].fillna(0)

## only keep columns, County, Year, Month, Human_Disease_Count
data_eric_13 = data_eric_13[["County", "Year", "Month", "Human_Disease_Count"]]

## sort the data by County, Year, Month
data_eric_13 = data_eric_13.sort_values(by=["County", "Year", "Month"])

## merge the two data based on County, Year, Month
data_merged = pd.merge(data_CA_13, data_eric_13, on=["County", "Year", "Month"], how="inner", suffixes=("_CA", "_Eric"))

## add a new column difference between Human_Disease_Count_CA and Human_Disease_Count_Eric, called Case_Difference
data_merged["Case_Difference"] = data_merged["Human_Disease_Count_CA"] - data_merged["Human_Disease_Count_Eric"]

data_merged.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/case_difference_between_two_sources.csv", index=False)