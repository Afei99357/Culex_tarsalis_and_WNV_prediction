import pandas as pd

# read a data from 2018 to 2022 from csv files in folder
df_2018 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/county_level_data_2018_2022/raw/"
                      "WNVPositiveAnimalData_2018.csv", index_col=False)
df_2019 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/county_level_data_2018_2022/raw/"
                        "WNVPositiveAnimalData_2019.csv", index_col=False)
df_2020 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/county_level_data_2018_2022/raw/"
                        "WNVPositiveAnimalData_2020.csv", index_col=False)
df_2021 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/county_level_data_2018_2022/raw/"
                        "WNVPositiveAnimalData_2021.csv", index_col=False)
df_2022 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/county_level_data_2018_2022/raw/"
                        "WNVPositiveAnimalData_2022.csv", index_col=False)

# combine the data to one dataframe
df = pd.concat([df_2018, df_2019, df_2020, df_2021, df_2022], ignore_index=True)

# drop the column "Unnamed: 3"
df = df.drop("Unnamed: 3", axis=1)

# strip the white space in column "County" and "Animal/Insect" and to lower case
df["County"] = df["County"].str.strip()
df["Animal/Insect"] = df["Animal/Insect"].str.strip()
df["County"] = df["County"].str.lower()


# add a column "Species" and fill in the value based on the value in column "Animal/Insect", if it is MOSQUITO, fill in Mosquito
# if it is HORSE, fill in Horse, anything else, fill in Bird
df["Species"] = df.apply(lambda x: "Mosquito" if x["Animal/Insect"] == "MOSQUITO" else "Horse" if x["Animal/Insect"] == "HORSE" else "Bird", axis=1)

# print unique values in Animal/Insect column
print(df["Animal/Insect"].unique())

# drop the column "Animal/Insect"
df = df.drop("Animal/Insect", axis=1)

# add columns "Year" and "Month" and fill in the value based on the value in column "Date"
df["Year"] = df["Date Collected"].str.split("/").str[2]
df["Month"] = df["Date Collected"].str.split("/").str[0]

# groupby County, Year, Month, Species, and get the count of each group
df = df.groupby(["County", "Year", "Month", "Species"]).size().reset_index(name="Count")

# pivot the table to make the Species as columns
df = df.pivot_table(index=["County", "Year", "Month"], columns="Species", values="Count").reset_index()

# if the county name is dewitt, change it to de witt, saint clair to st. clair
df["County"] = df["County"].str.replace("dewitt", "de witt")
df["County"] = df["County"].str.replace("saint clair", "st. clair")

# output the data to csv file
df.to_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/county_level_data_2018_2022_combine.csv", index=False)

# # drop column bird and horse
df_mosquitoes = df.drop(["Horse", "Mosquito"], axis=1)

# # drop nan values
df_mosquitoes = df_mosquitoes.dropna()

# # add county seat coordinates
df_county_seat = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/"
                             "Data/human_neuroinvasive_wnv_ebirds.csv", index_col=0)
# # state in Illinois
df_county_seat = df_county_seat[df_county_seat["State"] == "Illinois"]

# # only get the county seat coordinates and county name
df_county_seat = df_county_seat[["County", "Year", "FIPS", "County_Seat_Longitude", "County_Seat_Latitude",
                                 "Poverty_Estimate_All_Ages", "Population", "Land_Area_2010", "Avian Phylodiversity"]]

# # get unique pairs of county and county seat coordinates
df_county_seat = df_county_seat.drop_duplicates()

# # lower case the values in County and strip the white space
df_mosquitoes["County"] = df_mosquitoes["County"].str.lower()
df_mosquitoes["County"] = df_mosquitoes["County"].str.strip()
df_county_seat["County"] = df_county_seat["County"].str.lower()
df_county_seat["County"] = df_county_seat["County"].str.strip()

# convert the data type of Year to int
df_mosquitoes["Year"] = df_mosquitoes["Year"].astype(int)
df_county_seat["Year"] = df_county_seat["Year"].astype(int)

# # based on  the county name and year, merge the county seat coordinates to the df_mosquitoes
df_mosquitoes = pd.merge(df_mosquitoes, df_county_seat, on=["County", "Year"], how="left")

## output the data
df_mosquitoes.to_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/county_level_data_2018_2022/"
          "bird_illnois_2018_to_2022.csv")







