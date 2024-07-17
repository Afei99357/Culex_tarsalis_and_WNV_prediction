import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import re
import seaborn as sns

from keras.losses import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#
#
# #### convert bird species information to csv file ####
# # read a txt file into a list
# df = pd.read_fwf("/Users/ericliao/Desktop/WNV_project_files/birds_data/SpeciesList_edited.txt", encoding="ISO-8859-1", header=None)
# # remove the last two columns of df
# df = df.iloc[:, :-2]
# # remove the third and fourth columns of df
# df1 = df.drop(df.columns[[3, 4]], axis=1)
# # get the first column of df1 and turn to three columns
# df2 = df1[0].str.split(" ", n=2, expand=True)
# df1 = df1.drop(df1.columns[[0]], axis=1)
#
# # merge the two dataframes together where df2 is the first dataframe
# df3 = pd.merge(df2, df1, left_index=True, right_index=True)
#
# # rename the columns
# df3.columns = ["Seq", "AOU", "English_Common_Name", "French_Common_Name", "Spanish_Common_Name", "ORDER", "Family", "Genus", "Species"]
#
# df3.to_csv("/Users/ericliao/Desktop/WNV_project_files/birds_data/SpeciesList_edited.csv")
# #######################################################


# #### check us state infomation ####
#
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/US_county_info/us_county_boundaries.csv", sep=";")
#
# df_new = df[["Geo Point", "FIPS"]].to_csv("/Users/ericliao/Desktop/WNV_project_files/US_county_info/us_county_FIPS_3233.csv")
# ##################################


# ##### combine birds information all together based on state #####
# ####### read state bird files in the same folder and combine them all together  #######################
# # get the current working directory
# path = "/Users/ericliao/Desktop/WNV_project_files/birds_data/States"
# # get all the files in the current working directory
# all_files = glob.glob(os.path.join(path, "*.csv"))
# # create a list to store the dataframes
# li = []
# # loop through all the files
# for filename in all_files:
#     # read the file into a dataframe
#     df = pd.read_csv(filename, index_col=None, header=0)
#     # append a new column to the dataframe and give it the value of the name of the file
#     df["State"] = filename.split("/")[-1].split(".")[0]
#     # append the dataframe to the list
#     li.append(df)
#
# # concatenate all the dataframes in the list
# frame = pd.concat(li, ignore_index=True)
#
# # write the dataframe to a csv file
# frame.to_csv("/Users/ericliao/Desktop/WNV_project_files/birds_data/birds_counts_all_states_all_years.csv")
# ##############################################################################################


# ### based on AOU code, get the bird species name from species.csv file ####
# # read the birds_counts_all_states_all_years.csv file
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/birds_data/birds_counts_all_states_all_years.csv")
# # read the species.csv file
# df1 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/birds_data/SpeciesList_edited.csv")
# # based on the AOU code, get the bird species name from species.csv file and add it to each column in the birds_counts_all_states_all_years.csv file
# df["English_Common_Name"] = df["AOU"].map(df1.set_index("AOU")["English_Common_Name"])
# # write the dataframe to a csv file
# df.to_csv("/Users/ericliao/Desktop/WNV_project_files/birds_data/birds_counts_all_states_all_years_with_species_name.csv")
# ##############################################################################################


# ### reduce the unless columns in the birds_counts_all_states_all_years_with_species_name.csv file ####
# import pandas as pd
#
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/birds_data/birds_counts_combined_original.csv")
#
# # only keep the columns where the CommonName is StateNum, Year, AOU, SpeciesTotal, State, Species
# df1 = df[["StateNum", "Year", "AOU", "SpeciesTotal", "State", "Species"]]
#
# # rename Species column to CommonName
# df1 = df1.rename(columns={"Species": "CommonName"})
# # write the dataframe to a csv file
# df1.to_csv("/Users/ericliao/Desktop/WNV_project_files/Finalized_files/birds_counts_species_state_1966_to_2021.csv")
# ##############################################################################################


# ### based on Geographic Area column to combine two files together ####
# # read the population of 2010 to 2019
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/population_county_yearly/county_population_yearly_2010_2019.csv")
# # read the population of 2020 to 2021
# df1 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/population_county_yearly/county_population_yearly_2020_2021.csv")
# # combine the two dataframes together based on the Geographic Area column
# df2 = pd.merge(df, df1, on="Geographic Area", how="outer", suffixes=("_2010_2019", "_2020_2021"))
# # write the dataframe to a csv file
# df2.to_csv("/Users/ericliao/Desktop/WNV_project_files/population_county_yearly/county_population_yearly_2010_2021.csv")
# # ##############################################################################################


# # ### based on fips, merge two population files together ####
# # # read the population of 2011 to 2021
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Finalized_files/county_population_and_density_yearly_2011_2021.csv")
# # # read the population of 2000 to 2010
# df1 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/demographic/2000_2010_population.csv")
# # # combine the two dataframes together based on the fips column
# df2 = pd.merge(df, df1, on="fips", how="left", suffixes=("_2011_2021", "_2000_2010"))
# # # write the dataframe to a csv file
# df2.to_csv("/Users/ericliao/Desktop/WNV_project_files/Finalized_files/county_population_and_density_yearly_2000_2021.csv")
# # ##############################################################################################


### based on the county name and state name, find the FIPS code from the us_county_FIPS_3233.csv file ####
# read the county_population_yearly_2010_2021.csv file
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/population_county_yearly/county_population_yearly_2010_2021.csv")
# # read the state_name_and_code_info.csv file
# df1 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/population_county_yearly/state_name_and_code_info.csv")
# # add a new column to df
# df["state_code"] = ""
# # create a dictionary to store the state name and state code
# state_name_code_dict = {}
# # loop through the df1 dataframe
# for index, row in df1.iterrows():
#     # add the state name and state code to the dictionary
#     state_name_code_dict[row["state"]] = row["code"]
# # based on the state name, get the state code from the dictionary
# code_list = []
# for index, row in df.iterrows():
#     code_list.append(state_name_code_dict[row['state name']])
#
# df['state_code'] = code_list

# # read the county_state_information.csv file
# df2 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/US_county_info/county_state_information.csv")
# # add FIPS values to the df based on the df1 county name and state_code column and df2 name and state column
# # new_df = pd.merge(df, df2,  how='left', left_on=['county name', 'state_code'], right_on=['name', 'state'])
#

# ## find fips for poverty file
# # read the county_state_information.csv file
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/demographic/poverty/poverty_county_2000_2021.csv", index_col=0)
# df1 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/US_county_info/county_state_information.csv")
# # add FIPS values to the df based on the df1 county name and state_code column and df2 name and state column
# new_df = pd.merge(df, df1,  how='left', left_on=['Name', 'Postal Code'], right_on=['name', 'state'])
# # drop the duplicate columns
# new_df = new_df.drop(columns=['Postal Code', 'name'])
#
# # #rename Name to County Name, state to State Code
# new_df = new_df.rename(columns={"Name": "County Name", "state": "State Code"})
#
# # write the dataframe to a csv file
# new_df.to_csv("/Users/ericliao/Desktop/WNV_project_files/demographic/poverty/poverty_county_2000_2021_with_fips.csv")
# #############################################################################################


# ##### separate Geo point to two columns #####
# # # read the us_county_centroid_coordinates_with_fips.csv file
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Finalized_files/us_county_centroid_coordinates_with_fips.csv")
#
# # break the Geo Point column into two columns according to the comma
# df[['latitude', 'longitude']] = df['Geo Point'].str.split(',', expand=True)
#
# # remove the Geo Point column and index column
# df = df.drop(columns=['Geo Point', 'Unnamed: 0'])
#
# # write the dataframe to a csv file
# df.to_csv("/Users/ericliao/Desktop/WNV_project_files/Finalized_files/us_county_centroid_coordinates_with_fips_modify.csv")
#
# ##############################################################################################


# ### read nonhuman_wmv.csv file and us_county_centoroid csv file, merge them together based on the FIPS code ####
# # read the nonhuman_wmv.csv file
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/OSF_Storage_US/nonhuman_wnv_2003-2019.csv")
# # read the us_county_centroid_coordinates_with_fips_modify.csv file
# df1 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Finalized_files/us_county_centroid_separated_coordinates_with_fips.csv")
# # merge the two dataframes together based on the FIPS code
# df2 = pd.merge(df, df1, on="FIPS", how="left")
# # write the dataframe to a csv file
# df2.to_csv("/Users/ericliao/Desktop/WNV_project_files/Finalized_files/nonhuman_wnv_2003-2019_with_county_centroid_coordinates.csv")
#############################################

# ## scatter plot of the prediction result and the actual result
# import matplotlib.pyplot as plt
# # # read the prediction and actual result file
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/poke_around_data/NN_wnv_1st_layer_relu.csv")
# # # create a scatter plot
# plt.scatter(df["actual_mosq"], df["pred_mosq"])
# # # set the x and y labels
# plt.xlabel("actual mosquito count")
# plt.ylabel("predicted mosquito count")
# # # set the title
# plt.title("Actual vs Prediction for non-human WNV")
# plt.show()
########################################################################

# ### merge table 1 and table 2 together based on the FIPS code ###
# # # read the table 1 file
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Finalized_files/demographic/county_population_yearly_2010_2021_with_FIPS.csv")
# # # read the table 2 file
# df1 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Finalized_files/demographic/population_density_by_county.csv")
# # # get the list of columns from df
# col_list = df.columns.tolist()
# # # add "Density per square mile of land area" column to the col_list
# col_list.append("Density per square mile of land area")
#
# # # merge the two dataframes together based on the FIPS code and only keep the columns that are needed
# df2 = pd.merge(df, df1, on="fips", how="left")[col_list]
# # # write the dataframe to a csv file
# df2.to_csv("/Users/ericliao/Desktop/WNV_project_files/Finalized_files/demographic/county_population_and_density_yearly_2010_2021.csv")
# ##############################################################################################


# ## read birds file and output the file, which if the CommonName countains "American Crow"
# # read the birds file
# df_origin = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Finalized_files/birds_counts_species_state_1966_to_2021.csv")
# # filter the dataframe based on the CommonName column
# df = df_origin[df_origin["CommonName"].str.contains("American Crow")]
# # ### combine the counts in SpeciesTotal column together, if State and Year are the same,
# # # # then add the counts together and put the new counts in the SpeciesTotal column
# # # # read the file
# # drop the index column
# df = df.drop(columns=["Unnamed: 0"])
#
# # # # create a new dataframe to store the data which CommonName is American Crow and AOU is 4880
# df1 = df[df["CommonName"] == "American Crow"]
# # # # create a new dataframe to store the data which CommonName is not American Crow and AOU is not 4880
# df2 = df[df["CommonName"] != "American Crow"]
#
# # combine all the rows in df1 into one row if the State and Year are the same, and add the counts together
# # keep the CommonName and AOU columns the same
# df3 = df1.groupby(["StateNum", "State", "Year"]).agg({"CommonName": "first", "AOU": "first", "SpeciesTotal": "sum"}).reset_index()
# # combine all the rows in df2 into one row if the State and Year are the same, and add the counts together
# # keep the CommonName and AOU columns the same
# df4 = df2.groupby(["StateNum", "State", "Year"]).agg({"CommonName": "first", "AOU": "first", "SpeciesTotal": "sum"}).reset_index()
#
# # # # combine the two dataframes together based on the same State and Year
# df5 = pd.merge(df3, df4, on=["State", "Year", "StateNum"], how="outer")
#
# # give nan values in the SpeciesTotal column a value of 0
# df5["SpeciesTotal_x"] = df5["SpeciesTotal_x"].fillna(0)
# df5["SpeciesTotal_y"] = df5["SpeciesTotal_y"].fillna(0)
#
# # and add the counts together, and put the new counts in the SpeciesTotal column
# df5["SpeciesTotal"] = df5["SpeciesTotal_x"] + df5["SpeciesTotal_y"]
# # # give the value "American Crow" to the CommonName column
# df5["CommonName"] = "American Crow"
# df5["AOU"] = 4880
# # # drop the other columns except for the StateNum, Year, AOU, SpeciesTotal, State and CommonName
# df5 = df5.drop(columns=["CommonName_x", "AOU_x", "SpeciesTotal_x", "CommonName_y", "AOU_y", "SpeciesTotal_y"])
# # # order the columns
# df5 = df5[["StateNum", "State", "Year", "AOU", "CommonName", "SpeciesTotal"]]
#
# # get the list of birds species
# common_name_list = ["American Robin", "American Goldfinch", "Blue Jay", "European Starling", "House Sparrow", "House Finch"]
# # # create a new dataframe to store all the data in the df_origin dataframe which CommonName is in the common_name_list
# df6 = df_origin[df_origin["CommonName"].isin(common_name_list)]
# # drop the index column
# df6 = df6.drop(columns=["Unnamed: 0"])
# ## based on the SatetNum, Year, AOU, State and CommonName, add the counts together, and put the new counts in the SpeciesTotal column
# df7 = df6.groupby(["StateNum", "State", "Year", "AOU", "CommonName"]).agg({"SpeciesTotal": "sum"}).reset_index()
# df7 = df7[["StateNum", "State", "Year", "AOU", "CommonName", "SpeciesTotal"]]
#
# # # combine the two dataframes together
# df8 = pd.concat([df5, df7], axis=0)
#
# # # # write the dataframe to a csv file
# df8.to_csv("/Users/ericliao/Desktop/WNV_project_files/Finalized_files/birds_counts_species_state_1966_to_2021_7_species.csv")
# ##############################################################################################


# ### read disease count data and population data based on the year, FIPS
# # # read the disease count data
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Finalized_files/human_neuroinvasive_wnv_2000-2021_with_county_centroid_coordinates.csv", index_col=0)
# # # read the population data
# df1 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Finalized_files/county_population_and_density_yearly_2000_2021.csv", index_col=0)
# # # based on the FIPS and Year form df, look for the row in df1, and get the population and density
# # # # create a new column to store the population
# df["Population"] = 0
# # # # create a new column to store the density
# df["Density per square mile of land area"] = 0
# # # loop through df, and get the population and density based on the FIPS and Year
# for i in range(len(df)):
#     # get the FIPS and Year
#     fips = df.loc[i, "FIPS"]
#     year = df.loc[i, "year"]
#     # create column name
#     year_column_name = str(year) + "_population"
#     # get the row in df1 which has the same FIPS and Year
#     df2 = df1[(df1["fips"] == fips)]
#     # get the population and density
#     if len(df2) == 0:
#         population = 0
#         density = 0
#     else:
#         population = df2[year_column_name].values[0]
#         density = df2["Density per square mile of land area"].values[0]
#     # put the population and density in the Population and Density per square mile of land area columns
#     df.loc[i, "Population"] = population
#     df.loc[i, "Density per square mile of land area"] = density
#     print(i)
#
# # # # write the dataframe to a csv file
# df.to_csv("/Users/ericliao/Desktop/WNV_project_files/Finalized_files/disease_human_neuroinvasive_wnv_2000-2021_with_county_centroid_coordinates_with_population_and_density.csv")
# ##############################################################################################

# ### read the disease data, and combine birds information to each county based on state and year
# # # read the disease data
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Finalized_files/disease_human_neuroinvasive_wnv_2000-2021_with_population.csv", index_col=0)
# # # read the birds data
# df1 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Finalized_files/birds_counts_species_state_1966_to_2021_7_species.csv", index_col=0)
# # # create 7 new columns to store 7 different species of the birds count
# df["American Robin"] = 0
# df["American Goldfinch"] = 0
# df["Blue Jay"] = 0
# df["European Starling"] = 0
# df["House Sparrow"] = 0
# df["House Finch"] = 0
# df["American Crow"] = 0
# # # loop through df, and get the birds count based on the State and Year and bird CommonName from df1
# for i in range(len(df)):
#     # get the state and year
#     state = df.loc[i, "state"]
#     year = df.loc[i, "year"]
#     # get the rows in df1 which has the same State and Year
#     df2 = df1[(df1["State"] == state) & (df1["Year"] == year)]
#     # get the birds count based on the CommonName
#     for j in range(len(df2)):
#         common_name = df2["CommonName"].values[j]
#         species_total = df2["SpeciesTotal"].values[j]
#         df.loc[i, common_name] = species_total
#     print(i)
#
# # # # write the dataframe to a csv file
# df.to_csv("/Users/ericliao/Desktop/WNV_project_files/Finalized_files/disease_human_neuroinvasive_wnv_2000-2021_demographic_bird.csv")
# ##############################################################################################
#
# ### using birds count data from 2019 and 2021 to infer the birds count in 2020
# # # read the birds count data
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Finalized_files/disease_human_neuroinvasive_wnv_2000-2021_demographic_bird.csv", index_col=0)
# df.interpolate(method="linear", axis=0, inplace=True, limit_direction="both", limit=1)
# df.to_csv("/Users/ericliao/Desktop/WNV_project_files/Finalized_files/disease_human_neuroinvasive_wnv_2000-2021_demographic_bird_filled_2020.csv")
# ##############################################################################################

# # ### adding values to a column of current csv file based on the value of another column in a different csv file
# # # # read the disease data
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/demographic/poverty/poverty_county.csv")
# df1 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/demographic/poverty/est00all.csv")
# # create a new column to store the population
# df["2000_Poverty Estimate, All Ages"] = 0
# # loop through df, and get the population based on the state FIPS Code and County FIPS Code
# for i in range(len(df)):
#     # get the state FIPS Code and County FIPS Code
#     state_fips = df.loc[i, "State FIPS Code"]
#     county_fips = df.loc[i, "County FIPS Code"]
#     # get the row in df1 which has the same state FIPS Code and County FIPS Code
#     df2 = df1[(df1["State FIPS Code"] == state_fips) & (df1["County FIPS Code"] == county_fips)]
#     # get the population
#     if len(df2) == 0:
#         population = 0
#     else:
#         population = df2["Poverty Estimate All Ages"].values[0]
#     # put the population in the Population column
#     df.loc[i, "2000_Poverty Estimate, All Ages"] = population
#
# # # # write the dataframe to a csv file
# df.to_csv("/Users/ericliao/Desktop/WNV_project_files/demographic/poverty/poverty_county_2000_2021.csv")
# ##############################################################################################


# ### add column for poverty population for each county for each year
# # # read the disease data
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/"
#                  "human_neuroinvasive_wnv_rate_log_population.csv", index_col=0)
# # # read the poverty data
# df1 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/demographic/poverty/"
#                   "poverty_county_2000_2021_with_fips.csv", index_col=0)
#
# # # loop through df, and get the poverty population based on the FIPS and Year
# for i in range(len(df)):
#     # get the FIPS and Year
#     fips = df.loc[i, "FIPS"]
#     year = df.loc[i, "Year"]
#     # create column name
#     year_column_name = str(year) + "_Poverty Estimate, All Ages"
#     # get the row in df1 which has the same FIPS and Year
#     df2 = df1[(df1["fips"] == fips)]
#     # get the population and density
#     if len(df2) == 0:
#         continue
#     else:
#         if year_column_name not in df2.columns:
#             continue
#         poverty_population = df2[year_column_name].values[0]
#     # put the population and density in the Population and Density per square mile of land area columns
#     df.loc[i, "Poverty Estimate, All Ages"] = poverty_population
#     print(i)
#
# # # write the dataframe to a csv file
# df.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/human_neuroinvasive_wnv_rate_log_population_correct_poverty.csv")
# ##############################################################################################

# ### add column for land area for each county for each year
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Finalized_files/disease_human_neuroinvasive_wnv_2000-2021_bird_demographic.csv", index_col=0)
# # # # read the land area data
# df1 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/demographic/land area/land_area_each_county_2010.csv")
#
# # add column for land area
# df["Land Area 2010"] = 0
#
# # loop through df1, and get the land area based on the county name
# for i in range(len(df1)):
#     # get the county name
#     county_name = df1.loc[i, "Areaname"]
#     # get the land area
#     land_area = df1.loc[i, "LND110210D"]
#     # put the land area in the Land Area 2010 column
#     df.loc[df["county"] == county_name, "Land Area 2010"] = land_area
#     print(i)
#
# # # write the dataframe to a csv file
# df.to_csv("/Users/ericliao/Desktop/WNV_project_files/Finalized_files/disease_human_neuroinvasive_wnv_2000-2021_bird_demographic_land_area.csv")
# ##############################################################################################
#
# # # check the distribution of diseases count##
# import pandas as pd
# import matplotlib.pyplot as plt
#
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_01/data_files/disease_human_neuroinvasive_whole_year.csv")
#
# # get rows with SET == "VET"
# # df1 = df[df["SET"] == "VET"]
#
# # plot histogram on count column
# df.hist(column="WNV_Count", bins=100)
#
# # plot the max value
# plt.axvline(x=df["WNV_Count"].max(), color="red")
#
# # show the plot
#
# plt.show()
# ##############################################################################################

#
# ## for horse data, based on FIPS, adding County, State, Year, County_Centroid_Latitude, County_Centroid_Longitude, Land_Area_2010,
# # American_Robin, American_Goldfinch, Blue_Jay, European_Starling, House_Sparrow, House_Finch, American_Crow, Land_Change_Count_Since_1992,
# # Land_Use_Class and Processed_Flag_Land_Use columns
# import pandas as pd
# import numpy as np
#
# # read the horse data
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/horse_data/horse_disease_with_0_case.csv", index_col=0)
#
# # read the FIPS, bird data and land use data
# df1 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_01/data_files/"
#                   "disease_human_neuroinvasive_whole_year.csv", index_col=0)
#
# # #add County, State, Year, County_Centroid_Latitude, County_Centroid_Longitude, Land_Area_2010,
# # #American_Robin, American_Goldfinch, Blue_Jay, European_Starling, House_Sparrow, House_Finch, American_Crow,
# # #Land_Change_Count_Since_1992, Land_Use_Class and Processed_Flag_Land_Use columns for df
# df["County"] = ""
# df["State"] = ""
# df["County_Centroid_Latitude"] = 0
# df["County_Centroid_Longitude"] = 0
# df["Land_Area_2010"] = 0
# df["American_Robin"] = 0
# df["American_Goldfinch"] = 0
# df["Blue_Jay"] = 0
# df["European_Starling"] = 0
# df["House_Sparrow"] = 0
# df["House_Finch"] = 0
# df["American_Crow"] = 0
# df["Land_Change_Count_Since_1992"] = 0
# df["Land_Use_Class"] = ""
# df["Processed_Flag_Land_Use"] = ""
# df["Poverty_Estimate_All_Ages"] = 0
#
# # based on FIPS in df and df1, get values for each row in df
# for i in range(len(df)):
#     # get the FIPS
#     fips = df.loc[i, "FIPS"]
#     # get the row in df1 which has the same FIPS
#     df2 = df1[(df1["FIPS"] == fips)]
#     # get the County, State, Year, County_Centroid_Latitude, County_Centroid_Longitude, Land_Area_2010,
#     # American_Robin, American_Goldfinch, Blue_Jay, European_Starling, House_Sparrow, House_Finch, American_Crow,
#     # Land_Change_Count_Since_1992, Land_Use_Class and Processed_Flag_Land_Use values
#     if len(df2) == 0:
#         continue
#     else:
#         county = df2["County"].values[0]
#         state = df2["State"].values[0]
#         county_centroid_latitude = df2["County_Centroid_Latitude"].values[0]
#         county_centroid_longitude = df2["County_Centroid_Longitude"].values[0]
#         land_area_2010 = df2["Land_Area_2010"].values[0]
#         american_robin = df2["American_Robin"].values[0]
#         american_goldfinch = df2["American_Goldfinch"].values[0]
#         blue_jay = df2["Blue_Jay"].values[0]
#         european_starling = df2["European_Starling"].values[0]
#         house_sparrow = df2["House_Sparrow"].values[0]
#         house_finch = df2["House_Finch"].values[0]
#         american_crow = df2["American_Crow"].values[0]
#         land_change_count_since_1992 = df2["Land_Change_Count_Since_1992"].values[0]
#         land_use_class = df2["Land_Use_Class"].values[0]
#         processed_flag_land_use = df2["Processed_Flag_Land_Use"].values[0]
#         poverty_estimate_all_ages = df2["Poverty_Estimate_All_Ages"].values[0]
#     ## put the values in the columns
#     df.loc[i, "County"] = county
#     df.loc[i, "State"] = state
#     df.loc[i, "County_Centroid_Latitude"] = county_centroid_latitude
#     df.loc[i, "County_Centroid_Longitude"] = county_centroid_longitude
#     df.loc[i, "Land_Area_2010"] = land_area_2010
#     df.loc[i, "American_Robin"] = american_robin
#     df.loc[i, "American_Goldfinch"] = american_goldfinch
#     df.loc[i, "Blue_Jay"] = blue_jay
#     df.loc[i, "European_Starling"] = european_starling
#     df.loc[i, "House_Sparrow"] = house_sparrow
#     df.loc[i, "House_Finch"] = house_finch
#     df.loc[i, "American_Crow"] = american_crow
#     df.loc[i, "Land_Change_Count_Since_1992"] = land_change_count_since_1992
#     df.loc[i, "Land_Use_Class"] = land_use_class
#     df.loc[i, "Processed_Flag_Land_Use"] = processed_flag_land_use
#     df.loc[i, "Poverty_Estimate_All_Ages"] = poverty_estimate_all_ages
#
# # save the df to csv file
# df.to_csv("/Users/ericliao/Desktop/WNV_project_files/horse_data/horse_disease_without_weather_with_0_case.csv.csv")
# ########################################################################################################################


# ## ADD HORSE POPULATION data to horse_disease_without_weather.csv
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/horse_data/horse_disease_without_weather_with_0_case.csv.csv", index_col=0)
# # read horse population data
# df1 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/horse_data/horse_population_2003_and_2016.csv")
#
# # based on the state from df and df1, get the VALUES and give to df
# df["State_Land_Area"] = 0
# df["Population"] = 0
# df["Density_Per_Square_Mile_Of_Land_Area"] = 0
# for i in range(len(df)):
#     state = df.loc[i, "State"]
#     df2 = df1[(df1["State_Name"] == state)]
#     if len(df2) == 0:
#         continue
#     else:
#         state_land_area = df2["State_Land_Area"].values[0]
#         population = df2["Horse_total_2016"].values[0]
#         density_per_square_mile_of_land_area = df2["Density_Per_Square_Mile_Of_Land_Area"].values[0]
#     df.loc[i, "State_Land_Area"] = state_land_area
#     df.loc[i, "Population"] = population
#     df.loc[i, "Density_Per_Square_Mile_Of_Land_Area"] = density_per_square_mile_of_land_area
#
# # save the df to csv file
# df.to_csv("/Users/ericliao/Desktop/WNV_project_files/horse_data/horse_disease_without_weather_with_0_case.csv.csv")
# ########################################################################################################################


# ##### adding 0 cases in horse data based on the year
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/horse_data/horse_disease.csv")
#
# # find unique years and unique FIPS
# unique_years = df["Year"].unique()
# unique_fips = df["FIPS"].unique()
#
# # create a copy of df
# df1 = df.copy()
#
# # create a new df, for each county, if the year in unique_years is not in the df, add a row with 0 cases to df
# for fips in unique_fips:
#     df2 = df[(df["FIPS"] == fips)]
#     for year in unique_years:
#         if year not in df2["Year"].values:
#             df1 = df1.append({"FIPS": fips, "Year": year, "WNV_Count": 0, 'SET': 'VET'}, ignore_index=True)
#         else:
#             continue
#
# # save the df1 to csv file
# df1.to_csv("/Users/ericliao/Desktop/WNV_project_files/horse_data/horse_disease_with_0_case.csv")
#
# # ########################################################################################################################

#
# # scatter plot the longitude and latitude for disease data
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_02/human_neuroinvasive_and_horse_0_case_whole_year.csv")
# df = df.dropna()
# df = df[(df["County_Centroid_Latitude"] != 0) & (df["County_Centroid_Longitude"] != 0)]
#
# # # only choose horse data
# df = df[(df["SET"] == "VET")]
#
# # for each year, plot the scatter plot, and plot all subplots in one figure, and order by year
# # create a grid for subplots
# fig, axs = plt.subplots(4, 5, figsize=(20, 20))
# fig.subplots_adjust(hspace=0.5, wspace=0.5)
# unique_years = df["Year"].unique()
# # order the unique_years
# unique_years = np.sort(unique_years)
# for i in range(len(unique_years)):
#     df1 = df[(df["Year"] == unique_years[i])]
#     axs[i//5, i%5].scatter(df1["County_Centroid_Longitude"], df1["County_Centroid_Latitude"], s=df1["WNV_Count"]*2, c=df1["WNV_Count"], cmap="tab20b", alpha=0.5)
#     # add color bar and make it horizontal at the bottom
#     fig.colorbar(axs[i // 5, i % 5].scatter(df1["County_Centroid_Longitude"], df1["County_Centroid_Latitude"],
#                                             s=df1["WNV_Count"] * 2, c=df1["WNV_Count"], cmap="tab20b", alpha=0.5),
#                  ax=axs[i // 5, i % 5], orientation="horizontal", fraction=0.05, pad=0.05)
#
#     # change the range of the color bar
#     axs[i//5, i%5].set_title("WNV cases for horse in {}".format(unique_years[i]))
#     axs[i//5, i%5].set_xlabel("Longitude")
#     axs[i//5, i%5].set_ylabel("Latitude")
#     axs[i//5, i%5].set_xticklabels([])
#     axs[i//5, i%5].set_yticklabels([])
#     axs[i//5, i%5].set_xticks([-130, -100, -70], minor=True)
#     axs[i//5, i%5].set_yticks([20, 30, 40, 50], minor=True)
#     axs[i//5, i%5].grid(which="minor", color="white", linestyle='-', linewidth=0.5)
#     axs[i//5, i%5].grid(which="major", color="white", linestyle='-', linewidth=1)
#     axs[i//5, i%5].tick_params(axis='both', which='both', length=0)
#     axs[i//5, i%5].set_axisbelow(True)
#     axs[i//5, i%5].set_aspect("equal")

# ## remove the last three empty subplots for human data
# for i in range(1, 4):
#     fig.delaxes(axs[4, 5-i])

# ## remove the last three emtpy subplots for horse data
# for i in range(1, 4):
#     fig.delaxes(axs[3, 5-i])
#
# # ##save the figure
# plt.savefig("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_02/results/horse_plot/"
#             "horse_wnv_neuroinvasive_disease_all_years_123.png",
#             dpi=1200,
#             bbox_inches="tight")
#

### plot the scatter plot for each year
# unique_years = df["Year"].unique()
# for year in unique_years:
#     df1 = df[(df["Year"] == year)]
#     plt.scatter(df1["County_Centroid_Longitude"], df1["County_Centroid_Latitude"],
#                 s=df1["WNV_Count"]*10,
#                 c=df1["WNV_Count"],
#                 cmap="tab20b",
#                 alpha=0.5)
#     # add color bar
#     plt.colorbar()
#     # change the range of the color bar
#     plt.clim(0, 10)
#     plt.title("WNV cases for horse in {}".format(year))
#     plt.xlabel("Longitude")
#     plt.ylabel("Latitude")
#     plt.savefig("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_02/results/horse_plot/counties_has_wnv_horse_{}.png".format(year), dpi=300)
#     plt.close()
########################################################################################################################

#
# ## using the 2017 wnv count as the prediction result and 2018 data as the actual data to get the MSE
# # # read the data
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_02/human_neuroinvasive_and_horse_0_case_whole_year.csv")
# df = df.dropna()
#
# df = df[(df["SET"] == "VET")]
#
# # get the horse disease count data on 2017
# df_2017 = df[(df["Year"] == 2017)]
# disease_2017 = df_2017["WNV_Count"].values
#
# # get the horse disease count data on 2018
# df_2018 = df[(df["Year"] == 2018)]
# disease_2018 = df_2018["WNV_Count"].values
#
# # standard deviation of the disease 2018
# std = np.std(disease_2018)
#
# # calculate the MSE
# rmse = np.sqrt(((disease_2017-disease_2018)**2).mean())
#
# print(rmse**2)
# print(std)
# ########################################################################################################################

### add state abbreviation to the data
# # # read the data
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_02/human_neuroinvasive_and_horse_0_case_whole_year.csv", index_col=0)
# ## read the county and state abbreviation data
# df1 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/US_county_info/state_name_and_code_info.csv", index_col=False)
# df1 = df1[["state", "code"]]
#
# # # merge the data
# df = pd.merge(df, df1, left_on="State", right_on="state", how="left")
# # # drop the state column
#
# # # save the data
# df.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_07/human_neuroinvasive_and_horse_0_case_whole_year.csv")


# ### convert non float to float#####################
#
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_07/"
#                            "human_neuroinvasive_and_horse_0_case_whole_year.csv", index_col=0)
#
# df["Population"] = df["Population"].str.replace(",", "")
# df["State_Land_Area"] = df["State_Land_Area"].str.replace(",", "")
# df["Poverty_Estimate_All_Ages"] = df["Poverty_Estimate_All_Ages"].str.replace(",", "")
#
# ## convert population and Povert_Estimate_All_Ages column to float
# df["Population"] = df["Population"].astype(float)
# df["State_Land_Area"] = df["State_Land_Area"].astype(float)
# df["Poverty_Estimate_All_Ages"] = df["Poverty_Estimate_All_Ages"].astype(float)
#
# # # save the data
# df.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_07/"
#           "human_neuroinvasive_and_horse_0_case_whole_year_remove.csv")
########################################################################################################################


## plot the distribution of the disease count for each year by county
# # read the data
df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/aggregate_by_county/all_years/"
                 "mos_illinois_county_02_to_22.csv", index_col=0)

# get unique years
unique_years = df["Year"].unique()

# for horses
## remove year of 2000, 2001, 2002, 2020, 2021 from unique_years
unique_years = np.delete(unique_years, [0, 1, 2, 20, 21])

# #
# ##
# #get the horse disease count data for each year and x axis is the state and y axis is the count and plot in a 4 * 5 grid
# share y axis
fig, axs = plt.subplots(5, 5, figsize=(20, 15), sharey=True)

for i in range(len(unique_years)):

    #######################
    df1 = df[(df["Year"] == unique_years[i]) & (df["SET"] == "VET")]
    df1 = df1.groupby("State_Code").sum()
    # df1 = df1.sort_values(by="WNV_Count", ascending=False)
    axs[i//5, i%5].bar(df1.index, df1["WNV_Count"])
    axs[i//5, i%5].set_title("WNV cases for horse in {}".format(unique_years[i]))
    axs[i//5, i%5].set_xlabel("State")
    axs[i//5, i%5].set_ylabel("WNV Count")
    axs[i//5, i%5].set_xticklabels(df1.index, rotation=90)
    axs[i//5, i%5].tick_params(axis='x', which='major', labelsize=8)
    axs[i//5, i%5].grid(which="major", color="white", linestyle='-', linewidth=1)
    axs[i//5, i%5].set_axisbelow(True)
    axs[i//5, i%5].set_facecolor("lightgrey")
    axs[i//5, i%5].set_yscale("log")
    axs[i//5, i%5].set_ylim(1, 1000)

# add title to the figure
fig.suptitle("WNV cases for horse in each state from 2003 to 2019", fontsize=20)
# the distance between the subplots
fig.subplots_adjust(hspace=1, wspace=0.5)
#
# # ### for human
# # # # ## remove the last three empty subplots
# # for i in range(1, 4):
# #     fig.delaxes(axs[4, 5-i])
#
# ### for horses###
# # remove the last row of subplots
# for i in range(1, 6):
#     fig.delaxes(axs[4, 5-i])
# # # ## remove the last three empty subplots
# for i in range(1, 4):
#     fig.delaxes(axs[3, 5-i])
# ####
#
# ##save the figure
# plt.savefig("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_07/results/horse_plot/horse_combine_all_years.png", dpi=1200, bbox_inches="tight")

# # # get the horse disease count data for each year and x axis is the state and y axis is the count
# for i in unique_years:
#     df1 = df[(df["Year"] == i) & (df["SET"] == "VET")]
#     df1 = df1.groupby("State_Code").sum()
#     # df1 = df1.sort_values(by="WNV_Count", ascending=False)
#     plt.bar(df1.index, df1["WNV_Count"])
#     plt.title("WNV cases for horse in {} by state".format(i))
#     plt.xlabel("State")
#     plt.ylabel("WNV Count")
#     plt.xticks(rotation=90)
#     plt.savefig("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_07/results/horse_plot/states_has_wnv_horse_{}.png".format(i), dpi=300)
#     plt.close()

#######################################################################################################################

# ######## get the corvariance between horse and human cases ########
# # # read the data
# data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_07/"
#                            "human_neuroinvasive_and_horse_0_case_whole_year.csv", index_col=0)
#
# # find Counties in southern california
# southern_california_counties = ["Los Angeles", "San Diego", "Orange", "Riverside", "San Bernardino", "Kern", "Ventura",
#                                         "Santa Barbara", "San Luis Obispo", "Imperial"]

# data = data[data["County"].isin(southern_california_counties) | data["State"].isin(['North Dakota', 'South Dakota', 'Colorado'])]

# ## get the unique FIPS
# unique_fips = data["FIPS"].unique()
#
# ## remove data contains year 2000, 2001, 2002, 2019, 2020, 2021
# data = data[data["Year"] != 2000]
# data = data[data["Year"] != 2001]
# data = data[data["Year"] != 2002]
# data = data[data["Year"] != 2019]
# data = data[data["Year"] != 2020]
# data = data[data["Year"] != 2021]
#
# # get horse data
# horse_data = data[data["SET"] == "VET"]
# # get human data
# human_data = data[data["SET"] == "HUMAN"]
#
# # merge the horse and human data by year and FIPS to keep only results has the same year and FIPS
# horse_human_data = pd.merge(horse_data, human_data, on=["Year", "FIPS"], how="inner")
#
# # plot 3d scatter plot, x axis is the horse count, y axis is the human count, z axis is the year
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(horse_human_data["WNV_Count_x"], horse_human_data["WNV_Count_y"], horse_human_data["Year"])
# ax.set_xlabel("Horse WNV Count")
# ax.set_ylabel("Human WNV Count")
# ax.set_zlabel("Year")
# # set z axis to each year
# ax.set_zticks(horse_human_data["Year"].unique())
# ax.legend()
# ax.set_title("Correlation between horse and human WNV cases")
# plt.show()
# ###############################
# #
#
# ### get the county seat coordinates for each county##
# # # read the data
# data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_13_to_2023_02_20/dataset/human_neuroinvasive_no_weather.csv", index_col=0)
#
# # # get the county seat data
# county_seat_data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/US_county_info/USA_county_seats.csv", index_col=False)
#
# ## remove space in the county_seat_data
# county_seat_data["State_Code"] = county_seat_data["State_Code"].str.strip()
# county_seat_data["County"] = county_seat_data["County"].str.strip()
#
# # # merge two file based on the State_Code and County
# df_merge = pd.merge(data, county_seat_data, left_on=["State_Code", "County"],
#                     right_on=["State_Code", "County"],
#                     how="left", suffixes=("", "_y"))
#
# # # save the data
# df_merge.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_13_to_2023_02_20/dataset/human_neuroinvasive_no_weather_with_county_seat.csv")
# #######################################################################################################################
#
#
# ######### modify the coordinates for county seat coordinates
# ## read the nonhuamn data from the csv file
# human_neural_df = pd.read_csv(
#     "/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_13_to_2023_02_20/dataset/"
#     "human_neuroinvasive_no_weather_with_county_seat.csv"
# )
#
# ## read the modify list from the csv file
# modify_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/US_county_info/county_seat_coordinates_modify_change.csv", index_col=0)
#
# # loop through modify_df, for each row, find the corresponding row based on the county and state in human_neural_df,
# # and change the County_Seat_Latitude and County_Seat_Longitude to the County_Centroid_Latitude and County_Centroid_Longitude in human_neural_df
# for index, row in modify_df.iterrows():
#     # get the county and state
#     county = row["County"]
#     state = row["State"]
#     # get the row index in human_neural_df
#     row_index = human_neural_df[(human_neural_df["County"] == county) & (human_neural_df["State"] == state)].index
#     # change the County_Seat_Latitude and County_Seat_Longitude to the County_Centroid_Latitude and County_Centroid_Longitude
#     human_neural_df.loc[row_index, "County_Seat_Latitude"] = human_neural_df.loc[row_index, "County_Centroid_Latitude"]
#     human_neural_df.loc[row_index, "County_Seat_Longitude"] = human_neural_df.loc[row_index, "County_Centroid_Longitude"]
#
# # save the data
# human_neural_df.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_13_to_2023_02_20/dataset/human_neuroinvasive_no_weather_with_county_seat_modify.csv")
# #######################################################################################################################
#
# # # adding  non-neural invasive human data
# # # read the non nerual data
# non_nerual_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/OSF_Storage_US_disease_data/"
#                    "human_nonneuroinvasive_wnv_1999-2020.csv", index_col=False)
#
# # # read the nerual data
# nerual_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_13_to_2023_02_20"
#                         "/dataset/human_neuroinvasive_with_extreme_weather_with_county_seat_modify.csv", index_col=False)
#
#
#
# ## merge the two data based on FIPS, Year, and SET
# df_merge = pd.merge(nerual_df, non_nerual_df, on=["FIPS", "Year", "SET"], how="left", suffixes=("", "_y"))
#
# df_merge_human_only = df_merge[df_merge['SET'] == 'HUMAN']
#
# # # save the data
# df_merge_human_only.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_13_to_2023_02_20/dataset/human_only_all_disease_data.csv", index=False)
# #######################################################################################################################

# ## calculate the variance for each fdature in the data
# # # read the data
# data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_13_to_2023_02_20/"
#                    "dataset/human_neuroinvasive_with_extreme_weather_with_county_seat_modify.csv", index_col=False)
#
# # drop rows contains nan
# data = data.dropna()
#
# # find Counties in southern california
# southern_california_counties = ["Los Angeles", "San Diego", "Orange", "Riverside", "San Bernardino", "Kern", "Ventura",
#                                 "Santa Barbara", "San Luis Obispo", "Imperial"]
#
# data = data[data["County"].isin(southern_california_counties) | data["State"].isin(['North Dakota', 'South Dakota', 'Colorado'])]
#
# # only on horse
# # data = data[data["SET"] == "VET"]
# # only on human
# data = data[data["SET"] == "HUMAN"]
#
# # remove the columns that are not needed and categorical data
# data = data.drop(["FIPS", "County", "State", "State_Code", "Year",
#                   "Land_Change_Count_Since_1992", "Land_Use_Class", "WNV_Count",
#                   "County_Centroid_Latitude", "County_Centroid_Longitude", "County_Seat_Latitude",
#                   "County_Seat_Longitude", "County_Seat", "Processed_Flag_Land_Use", 'SET',
#                   'Poverty_Estimate_All_Ages'
#                   ], axis=1)
#
# ## get the column u10_Jan and column swvl1_Dec index
# column_u10_Jan_index = data.columns.get_loc("u10_Jan")
# column_swvl1_Dec_index = data.columns.get_loc("swvl1_Dec")
#
# ## DROP the columns between column_u10_Jan and column_swvl1_Dec includes column_u10_Jan and column_swvl1_Dec
# data = data.drop(data.columns[column_u10_Jan_index:column_swvl1_Dec_index + 1], axis=1)
#
# ## treat nan as 0
# data = data.fillna(0)
#
# # # # display the variance for each feature box-and-whisker plot and save them in png files
# # # # get the column names
# column_names = data.columns
# # # # loop through each column
# for column_name in column_names:
#     # # # # get the data for the column
#     # if column_name == "sf_acc_Oct_to_Aug":
#     #     breakpoint()
#     column_data = data[column_name]
#     # # # create the box-and-whisker plot and in a horizontal orientation
#     plt.boxplot(column_data, vert=True)
#
#     # add standard deviation to the plot and set the text on the right side of the boxplot
#     plt.text(0.65, 0.6, "Standard Deviation: " + str(round(column_data.std(), 5)), horizontalalignment='left',
#              fontsize=7, transform=plt.gca().transAxes, c="red")
#     # add mean to the plot and set the text below the variance
#     plt.text(0.65, 0.5, "Mean: " + str(round(column_data.mean(), 5)), horizontalalignment='left',
#              fontsize=7, transform=plt.gca().transAxes, c="red")
#     # add median to the plot and set the text below the mean
#     plt.text(0.65, 0.4, "Median: " + str(round(column_data.median(), 5)), horizontalalignment='left',
#              fontsize=7, transform=plt.gca().transAxes, c="red")
#
#     # # # set the title
#     plt.title(column_name)
#
#     # # # save the figure
#     plt.savefig("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_13_to_2023_02_20/plots/"
#                 "boxplots_sc_sd_nd/" + column_name + ".png", dpi=300)
#     # # # close the figure
#     plt.close()
#
#
# # ## create a figure with the number of subplots equal to the number of features, which has 5 columns and (number of features / 5) rows
# # # # # # get the number of features
# # num_features = len(data.columns)
# # # # # # get the number of rows
# # num_rows = int(math.ceil(num_features / 5))
# # # # # # create the figure
# # fig, axes = plt.subplots(num_rows, 5, figsize=(20, 20))
# # # # # # get the column names
# # column_names = data.columns
# # # # # # loop through each column
# # for column_name in column_names:
# #     # # # get the data for the column
# #     column_data = data[column_name]
# #     # # # get the index of the column
# #     column_index = column_names.get_loc(column_name)
# #     ## # get the row and column index for the subplot
# #     row_index = int(column_index / 5)
# #     column_index = column_index % 5
# #     # # # plot the box-and-whisker plot
# #     axes[row_index, column_index].boxplot(column_data)
# #     # # # set the title
# #     axes[row_index, column_index].set_title(column_name)
# # # # # # show the figure
# # plt.show()
# #
# # #######################################################################################################################
#
#
#
# ## calculate the correlation matrix #######################################
# # load data
# data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/"
#                    "human_neuroinvasive_wnv_rate_log_population_correct_poverty.csv", index_col=0)
#
# # find Counties in southern california
# southern_california_counties = ["Los Angeles", "San Diego", "Orange", "Riverside", "San Bernardino", "Kern", "Ventura",
#                                 "Santa Barbara", "San Luis Obispo", "Imperial"]
#
# # data = data[data["County"].isin(southern_california_counties) | data["State"].isin(['North Dakota', 'South Dakota', 'Colorado'])]
#
# # # drop columns that are not features and drop target
# data = data.drop(["Year", "FIPS", "County", "State", "State_Land_Area", "State_Code", "Land_Change_Count_Since_1992",
#                   "Land_Use_Class", "County_Centroid_Latitude", "County_Centroid_Longitude", "County_Seat_Latitude",
#                   "County_Seat_Longitude", "County_Seat", "Processed_Flag_Land_Use", 'SET',
#                   'Poverty_Estimate_All_Ages', "Population",
#                   "WNV_Rate_Neural_With_All_Years",
#                   # "WNV_Rate_Neural_Without_99_21",
#                   # "WNV_Rate_Non_Neural_Without_99_21",
#                   "State_Horse_WNV_Rate", "WNV_Rate_Non_Neural_Without_99_21_log",
#                   "WNV_Rate_Neural_Without_99_21_log"# target column
#                   ], axis=1)
#
# ### drop monthly weather data block #######################
# ## get the column u10_Jan and column swvl1_Dec index
# column_u10_Jan_index = data.columns.get_loc("u10_Jan")
# column_swvl1_Dec_index = data.columns.get_loc("swvl1_Dec")
#
# ## DROP the columns between column_u10_Jan and column_swvl1_Dec includes column_u10_Jan and column_swvl1_Dec
# data = data.drop(data.columns[column_u10_Jan_index:column_swvl1_Dec_index + 1], axis=1)
# ################################################################
#
# data = data.dropna()
#
# ## Calculate the correlation matrix
# corr_matirx = data.corr().to_numpy()
#
# # create a dict to store the column names, index using numbers, start from 0, key use the name of column name
# column_names_dict = {}
# for i in range(len(data.columns)):
#     column_names_dict[i] = data.columns[i]
#
# labels = list(column_names_dict.keys())
#
# df = pd.DataFrame(corr_matirx, index=labels, columns=labels)
#
# fig, ax = plt.subplots(figsize=(40, 20))
#
# ## plot the correlation matrix in a heatmap
# sns.heatmap(df, annot=False, cmap='coolwarm')
#
# plt.text(19.5, -0.5, "Legend:", fontsize=25, rotation=0, c="black")
# ## add text as legend, for each row is a key and value in the dictionary
# for i in range(len(labels)):
#     plt.text(19.5, i*5/10, list(column_names_dict.items())[i], fontsize=20, rotation=0, c="black")
#
# # # make the color bar font size
# cbar = ax.collections[0].colorbar
# cbar.ax.tick_params(labelsize=25)
#
# # add title
# plt.title("Pearson Correlation Matrix Heatmap", fontsize=30)
#
# # x axis tick font
# plt.xticks(fontsize=25)
#
# # y axis tick font
# plt.yticks(fontsize=25)
#
# ## display the plot
# plt.savefig("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_13_to_2023_02_20/plots/pearson_correlation_matrix_heatmap.png", dpi=300)
# # ######################################################################################################################
#

# ### to normalize the WNV_Count by county population, then remove population and density features
# # # read the data
# data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/"
#                    "human_neuroinvasive_with_extreme_weather_with_county_seat_modify.csv", index_col=False)
#
# # # only choose human
# data_human = data[data['SET'] == "HUMAN"]
# data_horse = data[data['SET'] == "VET"]
#
# # # create a new column WNV_Rate
# data_human["Human_WNV_Rate"] = data_human["WNV_Count"] / data_human["Population"]
#
# data_horse["State_Horse_WNV_Rate"] = data_horse["WNV_Count"] / data_horse["Population"]
#
# # only keep State_Horse_WNV_Rate, FIPS and Year columns
# data_horse = data_horse[["State_Horse_WNV_Rate", "FIPS", "Year"]]
#
# # merge data_human and data_horse by FIPS and Year
# df_merge = pd.merge(data_human, data_horse, on=["FIPS", "Year"], how="left")
#
# # # drop columns WNV_Count and Population and density
# df_merge = df_merge.drop([
#     "WNV_Count",
#     "Population",
#     "Density_Per_Square_Mile_Of_Land_Area"
# ], axis=1)
#
# # # output the data
# df_merge.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/human_neuroinvasive_wnv_rate.csv", index=False)
# #######################################################################################################################
# #
#
# ### prepare the neural and non-neural invasive dataset###
# df_wnv_all = pd.read_csv('/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/'
#                          'human_neuroinvasive_with_extreme_weather_with_county_seat_modify.csv', index_col=False)
# df_neural = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/OSF_Storage_US_disease_data/human_neuroinvasive_wnv_2000-2021.csv", index_col=False)
# df_neural = df_neural[df_neural['Year'] < 2021]
#
# df_non_neural = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/OSF_Storage_US_disease_data/human_nonneuroinvasive_wnv_1999-2020.csv", index_col=False)
# df_non_neural = df_non_neural[df_non_neural['Year'] > 1999]
#
# ## get FIPS, Year and Population data from df_wnv_all
# df_wnv_population = df_wnv_all[["FIPS", "Year", "Population"]]
#
# # merge two df together without losing data from either side
# df_merge = pd.merge(df_neural, df_non_neural, on=["FIPS", "Year"], how="outer", suffixes=("_Neural", "_Non_Neural"))
#
# ## merge df_merge and df_wnv_population
# df_merge = pd.merge(df_merge, df_wnv_population, on=["FIPS", "Year"], how="left")
#
# ## drop columns County-non-neural adn State_Non_Neural
# df_merge = df_merge.drop(["County_Non_Neural", "State_Non_Neural"], axis=1)
#
# ## fill 0 for missing value in Count column
# df_merge["Count_Non_Neural"] = df_merge["Count_Non_Neural"].fillna(0)
#
# ## add column WNV_Rate_Non_Neural and WNV_Rate_Neural
# df_merge["WNV_Rate_Neural"] = df_merge["Count_Neural"] / df_merge["Population"]
# df_merge["WNV_Rate_Non_Neural"] = df_merge["Count_Non_Neural"] / df_merge["Population"]
#
# # df_merge.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/human_neuroinvasive_non_neuroinvasive_wnv_2000_2020_combine_fill_na.csv", index=False)
#
# grouped_df = df_merge.groupby(["Year", "FIPS"])
#
# neural_counts = grouped_df["Count_Neural"].sum()
# non_neural_counts = grouped_df["Count_Non_Neural"].sum()
#
# # neural_counts = grouped_df["WNV_Rate_Neural"].sum()
# # non_neural_counts = grouped_df["WNV_Rate_Non_Neural"].sum()
#
# ## correlation coefficient
# corr_coef = neural_counts.corr(non_neural_counts)
#
# ## calculate covariance
# covar = neural_counts.cov(non_neural_counts)
#
# print(f"Correlation coefficient: {corr_coef}")
# print(f"Covariance: {covar}")
#
# plt.scatter(neural_counts, non_neural_counts)
# plt.xlabel("Neural Invasive Count")
# plt.ylabel("Non-Neural Invasive Count")
# plt.xscale("log")
# plt.yscale("log")
# plt.title(f"Neural Invasive WNV Count vs. Non-Neural Invasive Count\n Correlation Coefficient: {corr_coef}")
#
# plt.savefig("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_13_to_2023_02_20/plots/wnv_count_neural_vs_non_neural_corr_coef.png", dpi=300)
#
# plt.show()
##############################################################################################

# # # calculate correlation coefficient horse wnv rate and human wnv rate
#
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/"
#                  "human_neuroinvasive_wnv_rate.csv", index_col=False)
#
# df = df.dropna()
#
# grouped_df = df.groupby(["Year", "FIPS"])
#
#
# human_rate = grouped_df["Human_WNV_Rate"].sum()
# horse_rate = grouped_df["State_Horse_WNV_Rate"].sum()
#
# ## correlation coefficient
# corr_coef = human_rate.corr(horse_rate)
#
# ## calculate covariance
# covar = human_rate.cov(horse_rate)
#
# print(f"Correlation coefficient: {corr_coef}")
# print(f"Covariance: {covar}")
#
# plt.scatter(human_rate, horse_rate)
# plt.xlabel("Human WNV Rate")
# plt.ylabel("Horse WNV Rate")
# # # plot in log space for both axis
# plt.xscale("log")
# plt.yscale("log")
#
# plt.title(f"Human WNV Rate vs. Horse WNV Rate\n Correlation Coefficient: {corr_coef}")
#
# plt.savefig("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_13_to_2023_02_20/plots/wnv_rate_human_vs_horse_corr_coef.png", dpi=300)
#
# plt.show()
# #################################################################################################


###### loading WNV dataset and calculate correlation coefficient for each pair of features
# ## and plot the scatter plot for each pair
# # load data
# data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/"
#                    "human_neuroinvasive_wnv_rate_log_population_correct_poverty.csv", index_col=0)
#
# # find Counties in southern california
# southern_california_counties = ["Los Angeles", "San Diego", "Orange", "Riverside", "San Bernardino", "Kern", "Ventura",
#                                 "Santa Barbara", "San Luis Obispo", "Imperial"]
#
# # data = data[data["County"].isin(southern_california_counties) | data["State"].isin(['North Dakota', 'South Dakota', 'Colorado'])]
#
# # # drop columns that are not features and drop target
# data = data.drop(["Year", "FIPS", "County", "State", "State_Land_Area", "State_Code", "Land_Change_Count_Since_1992",
#                   "Land_Use_Class", "County_Centroid_Latitude", "County_Centroid_Longitude", "County_Seat_Latitude",
#                   "County_Seat_Longitude", "County_Seat", "Processed_Flag_Land_Use", 'SET',
#                   'Poverty_Estimate_All_Ages', "Population",
#                   "WNV_Rate_Neural_With_All_Years",
#                   # "WNV_Rate_Neural_Without_99_21",
#                   # "WNV_Rate_Non_Neural_Without_99_21",
#                   "State_Horse_WNV_Rate", "WNV_Rate_Non_Neural_Without_99_21_log",
#                   "WNV_Rate_Neural_Without_99_21_log"# target column
#                   ], axis=1)
#
# ### drop monthly weather data block #######################
# ## get the column u10_Jan and column swvl1_Dec index
# column_u10_Jan_index = data.columns.get_loc("u10_Jan")
# column_swvl1_Dec_index = data.columns.get_loc("swvl1_Dec")
#
# ## DROP the columns between column_u10_Jan and column_swvl1_Dec includes column_u10_Jan and column_swvl1_Dec
# data = data.drop(data.columns[column_u10_Jan_index:column_swvl1_Dec_index + 1], axis=1)
# ################################################################
#
# # # remove rows with missing values
# data = data.dropna()
#
# # get the list of column names
# col_names = list(data.columns)
#
# # for loop to calculate correlation coeeficient between each pair of features
# for i in range(len(col_names)):
#     for j in range(i+1, len(col_names)):
#         feature_1 = data[col_names[i]]
#         feature_2 = data[col_names[j]]
#         corr_coef = feature_1.corr(feature_2)
#         print(f"Correlation coefficient between {col_names[i]} and {col_names[j]}: {corr_coef}")
#
#         # plot scatter plot
#         plt.scatter(feature_1, feature_2)
#         plt.xlabel(col_names[i])
#         plt.ylabel(col_names[j])
#
#         plt.xscale("log")
#         plt.yscale("log")
#
#         plt.title(f"{col_names[i]} vs. {col_names[j]}\n Correlation Coefficient: {corr_coef}")
#
#         plt.savefig(f"/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_13_to_2023_02_20/"
#                     f"plots/correlation_coefficient/feature_pair/scatter_plot_{corr_coef:.3f}_{col_names[i]}_vs_{col_names[j]}.png", dpi=300)
#         plt.show()
# ###############################################################################################



########### add two feature as log_WNV_Rate
#
# ### read the data
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/"
#                  "human_neuroinvasive_wnv_rate.csv", index_col=False)
#
# df_wnv_rate = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/"
#                           "human_neuroinvasive_non_neuroinvasive_wnv_rate_2000_2020.csv", index_col=False)
#
#
# # merge df and df_wnv_rate by Year and FIPS
# df_merge = pd.merge(df, df_wnv_rate, on=["Year", "FIPS"], how="left")
#
# # df_merge = df_merge.drop(["Human_WNV_Rate", "County_Neural", "State_Neural", "Location", "Count_Neural", "Count_Non_Neural",
# #                 "Population"])
#
# #rename WNV_Rate_Neural to WNV_Rate_Neural_Without_2021
# df_merge["WNV_Rate_Neural_Without_99_21"] = df_merge["WNV_Rate_Neural"]
# df_merge["WNV_Rate_Neural_With_All_Years"] = df_merge["Human_WNV_Rate"]
# df_merge["WNV_Rate_Non_Neural_Without_99_21"] = df_merge["WNV_Rate_Non_Neural"]
#
# df_merge = df_merge.drop(["WNV_Rate_Neural", "WNV_Rate_Non_Neural", "Human_WNV_Rate"], axis=1)
#
# ## add two column as WNV_Rate_Neural_Log and WNV_Rate_Non_Neural_Log and avoid log0
# df_merge["WNV_Rate_Neural_Without_99_21_log"] = np.log(df_merge["WNV_Rate_Neural_Without_99_21"] + 1)
# df_merge["WNV_Rate_Non_Neural_Without_99_21_log"] = np.log(df_merge["WNV_Rate_Non_Neural_Without_99_21"] + 1)
#
# df_merge.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/"
#                 "human_neuroinvasive_wnv_rate_log.csv")
# ########################################################################################



# #### calculate correlation coefficient for each pair of extrem weather by different year
# ## and plot the scatter plot for each pair
# #
# # load data
# data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/"
#                    "human_neuroinvasive_wnv_rate_log_population_correct_poverty.csv", index_col=0)
#
# # remove horse wnv rate column
# data.pop("State_Horse_WNV_Rate")
#
# # find Counties in southern california
#
# ### southern_california_counties = ["Los Angeles", "San Diego", "Orange", "Riverside", "San Bernardino", "Kern", "Ventura","Santa Barbara", "San Luis Obispo", "Imperial"]
#
# ### in FIPS
# southern_california_counties = [6037, 6073, 6059, 6065, 6071, 6029, 6111, 6083, 6079, 6025]
#
# data = data[data["FIPS"].isin(southern_california_counties) | data["State"].isin(['North Dakota', 'South Dakota', 'Colorado'])]
#
# # drop columns that are not features
# data = data.drop(["County", "State", "State_Code", "Land_Change_Count_Since_1992", "Land_Use_Class",
#                   "County_Centroid_Latitude", "County_Centroid_Longitude", "County_Seat_Latitude",
#                   "County_Seat_Longitude", "County_Seat", "Processed_Flag_Land_Use", 'SET', 'Poverty_Estimate_All_Ages',
#                   "WNV_Rate_Neural_With_All_Years", "WNV_Rate_Neural_Without_99_21", "WNV_Rate_Non_Neural_Without_99_21"
#                           ], axis=1)
#
# ### drop monthly weather data block #######################
# ## get the column u10_Jan and column swvl1_Dec index
# column_u10_Jan_index = data.columns.get_loc("u10_Jan")
# column_swvl1_Dec_index = data.columns.get_loc("swvl1_Dec")
#
# ## DROP the columns between column_u10_Jan and column_swvl1_Dec includes column_u10_Jan and column_swvl1_Dec
# data = data.drop(data.columns[column_u10_Jan_index:column_swvl1_Dec_index + 1], axis=1)
# ################################################################
#
# # # remove rows with missing values
# data = data.dropna()
#
# # get the list of column names
# col_names = list(data.columns)
# ## remove Year and FIPS from the col_names
# col_names.remove("Year")
# col_names.remove("FIPS")
#
# # for loop to calculate correlation coeeficient between each pair of features
# for i in range(len(col_names)):
#     for j in range(i+1, len(col_names)):
#         feature_1 = data[col_names[i]]
#         feature_2 = data[col_names[j]]
#         corr_coef = feature_1.corr(feature_2)
#         print(f"Correlation coefficient between {col_names[i]} and {col_names[j]}: {corr_coef}")
#
#         # plot scatter plot
#         plt.scatter(feature_1, feature_2)
#         plt.xlabel(col_names[i])
#         plt.ylabel(col_names[j])
#         # plt.xscale("log")
#         plt.yscale("log")
#         plt.title(f"{col_names[i]} vs. {col_names[j]}\n Correlation Coefficient: {corr_coef}")
#
#         plt.savefig(f"/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_13_to_2023_02_20/"
#                     f"plots/correlation_coefficient/feature_pair_only_ca_nd_sd_co/scatter_plot_{corr_coef:.3f}_{col_names[i]}_vs_{col_names[j]}.png", dpi=300)
#         plt.show()
# ##############################################################################################


#
# ## adding population to file################################
# ## read file
# data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/"
#                    "human_neuroinvasive_wnv_rate_log_population_correct_poverty.csv", index_col=0)
#
# population_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/demographic/population_with_fips_year.csv",
#                             index_col=False)
#
# ## merge by FIPS and Year
# data_merge = pd.merge(data, population_df, on=["FIPS", "Year"], how="left")
#
# ## convert Population to float
# data_merge["Population"] = data_merge["Population"].str.replace(",", "")
# data_merge["Population"] = data_merge["Population"].astype(float)
#
# data_merge["Poverty_Estimate_All_Ages"] = data_merge["Poverty_Estimate_All_Ages"].str.replace(",", "")
# data_merge["Poverty_Estimate_All_Ages"] = data_merge["Poverty_Estimate_All_Ages"].astype(float)
#
# ## create a new column called Poverty_Rate_All_Age
# data_merge["Poverty_Rate_Estimate_All_Ages"] = data_merge["Poverty_Estimate_All_Ages"] / data_merge["Population"]
#
# data_merge.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/"
#                   "human_neuroinvasive_wnv_rate_log_population_correct_poverty.csv")
# ########################################################################################################################

#
# ## #########################################################merge to get poverty of 2020 & 2021#########################################################
#
# df1 =  pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/demographic/poverty/poverty_county_2000_2021_with_fips.csv", index_col=0)
#
# df2 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/demographic/poverty/est20all.csv", index_col=False)
#
# df3 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/demographic/poverty/est21all.csv", index_col=False)
#
# df_merge = pd.merge(df1, df2, on=["State FIPS Code", "County FIPS Code", "State Code", "County Name"], how="left")
#
# df_merge = pd.merge(df_merge, df3, on=["State FIPS Code", "County FIPS Code", "State Code", "County Name"], how="left")
#
# df_merge.to_csv("/Users/ericliao/Desktop/WNV_project_files/demographic/poverty/poverty_county_2000_2021_with_fips_2020_2021.csv")
#
# ##################################################################################################################

## include avian_phylodiversity ######
# read data file
# data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/"
#                    "human_neuroinvasive_wnv_ebirds.csv", index_col=0)
#
# # read avian_phylodiversity file
# avian_data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/birds_data/estimates_of_avian_phylodiversity/"
#                          "avi_phylodiv_wnv_041822.csv", index_col=0)
#
# # # based on FIPS to merge avian_data to data
# data_merge = pd.merge(data, avian_data, how="left", left_on="FIPS", right_on="STCO_FIPS")
#
# data_merge.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/human_neuroinvasive_wnv_ebirds_avian_phylo.csv", index_label=True)
#
###############################################

#
# ## add FIPS, and some information to california weekly data
# # # read california data
# cali_data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/Data/"
#                    "California_wnv_count_weekly_2011_2012_modify_ebirds.csv", index_col=0)
#
# ## read disease data
# disease_data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/Data/human_neuroinvasive_wnv_ebirds.csv", index_col=0)
#
# disease_data = disease_data[["FIPS", "County", "State", "Year", "County_Seat_Latitude", "County_Seat_Longitude",
#                              "Population", 'Poverty_Rate_Estimate_All_Ages', "Land_Area_2010",
#                              "Avian Phylodiversity"]]
#
# disease_data = disease_data[(disease_data['Year'].isin([2011, 2012])) & (disease_data['State'] == "California")]
#
# # # drop duplicate rows in disease data
# disease_data = disease_data.drop_duplicates()
#
# # # merge data based on Year, County and State
# df_merge = pd.merge(cali_data, disease_data, how='left', on=["Year", "County", "State"])
#
# df_merge.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/Data/California_wnv_count_weekly_2011_2012_modify_ebirds_demographic.csv", index=True)
#
#############################################################
#
# ############### create a empty data frame to store all counties in california for each month and each year############
#
# # read file
# data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/Data/"
#                    "human_neuroinvasive_wnv_ebirds.csv", index_col=0)
#
# # get the data in year 2011 and 2012 and only in california
# data = data[(data['Year'].isin([2004, 2005, 2006, 2010, 2011, 2012])) & (data['State'] == "California")]
#
# # only get certain columns
# data = data[["FIPS", "County", "State", "Year", "County_Seat_Latitude", "County_Seat_Longitude", "Population", "Land_Area_2010", "Avian Phylodiversity", 'Poverty_Estimate_All_Ages']]
#
# # get the unique data based on FIPS
# data = data.drop_duplicates(subset=['FIPS', "Year"])
#
# data['County'] = data['County'].str.lower()
#
# # get the unique county
# unique_county = np.unique(data['County']).tolist()
#
# # create a empty dataframe with size of unique_county * 12 rows, where dataframe has three columns "County", "Year", and "Month"
# df_empty = pd.DataFrame(columns=["County", "Year", "Month"])
#
# # create a list to store the year
# year_list = [2004, 2005, 2006, 2010, 2011, 2012]
#
# # create a list to store the month
# month_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
#
# # create a list to store the county
# county_list = unique_county
#
# # add values to the df_empty
# for year in year_list:
#     for county in county_list:
#         for month in month_list:
#             df_empty = df_empty.append({"County": county, "Year": year, "Month": month}, ignore_index=True)
#
# # import data from cali weekly
# df_cali = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/disease_data_weekly_CA/final_file/"
#                       "California_wnv_count_weekly.csv", index_col=False)
#
# # choose data where onset_date is not nan or county is not nan or Human_Type is not nan
# df_cali = df_cali[(df_cali['onset_date'].notna()) & (df_cali['County'].notna()) & (df_cali['Human_Type'].notna())]
#
# ## wnnd
# df_cali_wnnd = df_cali[df_cali["Human_Type"] == 'WNND']
#
# ## wnf
# df_cali_wnf = df_cali[df_cali["Human_Type"] == 'WNF']
#
# # separate onset_date to year, month and day
# df_cali_wnnd['Year'] = pd.DatetimeIndex(df_cali_wnnd['onset_date']).year
# df_cali_wnnd['Month'] = pd.DatetimeIndex(df_cali_wnnd['onset_date']).month
# df_cali_wnf['Year'] = pd.DatetimeIndex(df_cali_wnf['onset_date']).year
# df_cali_wnf['Month'] = pd.DatetimeIndex(df_cali_wnf['onset_date']).month
#
# df_cali_wnnd.pop('onset_date')
# df_cali_wnf.pop('onset_date')
#
# # convert both county in df_empty, df_cali and data into lowercase
# df_empty['County'] = df_empty['County'].str.lower()
# df_cali_wnnd['County'] = df_cali_wnnd['County'].str.lower()
# df_cali_wnf['County'] = df_cali_wnf['County'].str.lower()
#
# # remove space in the beginning and end in df_cali column of County
# df_cali_wnnd["County"] = df_cali_wnnd["County"].str.strip()
# df_cali_wnf["County"] = df_cali_wnf["County"].str.strip()
#
# # group by df_cali by year, month and county
# df_cali_wnnd = df_cali_wnnd.groupby(["Year", "Month", "County"]).size().reset_index(name='Human_WNND_Count')
# df_cali_wnf = df_cali_wnf.groupby(["Year", "Month", "County"]).size().reset_index(name='Human_WNF_Count')
#
# # merge data to get Human_Disease_Count based on county, year, month
# df_merge = pd.merge(df_empty, df_cali_wnnd, how='left', on=["County", "Month", 'Year'])
# df_merge = pd.merge(df_merge, df_cali_wnf, how='left', on=["County", "Month", 'Year'])
#
# # fill 0 for the nan value in "Human_Count" column
# # df_merge['Human_WNND_Count'] = df_merge['Human_WNND_Count'].fillna(0)
#
# # merge data and df_merge
# df_merge = pd.merge(df_merge, data, how='left', on=['County', 'Year'])
#
# # output
# df_merge.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
#                       "add_0_for_no_wnv/cali_week_wnnd_wnf_multi_years.csv")

#############################################################################################################

#
# ######
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
#                  "add_0_for_no_wnv/cali_week_wnnd_wnf_all_features.csv", index_col=0)
#
# # fill nan with 0 in Human_WNND_count
# df['Human_WNND_Count'] = df['Human_WNND_Count'].fillna(0)
#
# #todo: predict WNF based on WNND
# # drop the rows where Human_WNF_Count is nan
# df = df.dropna(subset=['Human_WNF_Count'])
#
# # create a column called Total_WNV_Count and the values equal to Human_WNND_Count plus Human_WNF_Count
# df['Total_WNV_Count'] = df['Human_WNF_Count'] + df['Human_WNND_Count']
# df['Total_WNV_Rate'] = df['Total_WNV_Count'] / df['Population']
# # output
#
# df.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
#                  "add_0_for_no_wnv/cali_week_wnnd_wnf_all_features_sum_wnv.csv")
###################

## check how many unique counties in the dataset

# import the dataset
df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/combine_cdc_all_birds.csv", index_col=0)

# get unique values of column FIPS
unique_fips = set(df['FIPS'].tolist())

# get the size of unique_fips
print(len(unique_fips))


