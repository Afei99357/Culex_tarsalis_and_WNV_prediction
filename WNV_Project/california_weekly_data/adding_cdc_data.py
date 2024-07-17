import pandas as pd


## CDC Birds data###############################################
# # Load the dataset into a Pandas DataFrame
# data_bird = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/"
#                        "Dead_Bird_Denominator_Numerator_monthly.csv", index_col=0)
# # only California
# data_bird = data_bird[data_bird["State"] == "CA"]
#
# # load disease data
# data_disease = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
#                            "add_0_for_no_wnv/cali_week_wnnd_multi_years_all_features_mosquitoes.csv", index_col=0)
#
#
# data_merge = pd.merge(data_disease, data_bird, how='left', left_on=['Year', 'Month', 'FIPS'], right_on=['Year', 'Month', 'County'])
#
# # only choose the rows that have WNV_Count
# # data_merge = data_merge[data_merge["WNV_Count"] >= 0]
#
# # drop columns that are not features and drop target
# data_merge = data_merge.drop(["State_y", "County_y", "CorvidsReported", "CorvidsTested",
#                               "OtherReported", "OtherTested"], axis=1)
#
# # rename the columns
# data_merge = data_merge.rename(columns={'State_x': 'State', 'County_x': 'County'})
#
# # reset index
# data_merge = data_merge.reset_index(drop=True)
#
# # make a copy of column WNV_Corvid_Count and add WNV_NonCorvid_Count, name them corvids and other
# data_merge["Corvid_Count"] = data_merge["WNV_Corvid_Count"]
# data_merge["NonCorvid_Count"] = data_merge["WNV_NonCorvid_Count"]
#
# # fill nan with 0
# data_merge["Corvid_Count"] = data_merge["Corvid_Count"].fillna(0)
# data_merge["NonCorvid_Count"] = data_merge["NonCorvid_Count"].fillna(0)
#
# # add a column total_bird_WNV_Count by adding up the WNV_Count of Corvids and Other
# data_merge["Total_Bird_WNV_Count"] = data_merge["Corvid_Count"] + data_merge["NonCorvid_Count"]
#
# # if both WNV_Corvid_Count and WNV_NonCorvid_Count are Nan, then replace total_bird_WNV_Count with Nan
# data_merge.loc[(data_merge["WNV_Corvid_Count"].isna()) & (data_merge["WNV_NonCorvid_Count"].isna()), "Total_Bird_WNV_Count"] = None
#
# # drop the columns Corvid_Count and NonCorvid_Count
# data_merge = data_merge.drop(["Corvid_Count", "NonCorvid_Count"], axis=1)
#
# data_merge.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
#                   "add_0_for_no_wnv/cali_week_wnnd_multi_years_all_features_mosquitoes_bird.csv")

####################################################################


### CDC Horse data###############################################
# # Load the dataset into a Pandas DataFrame
data_horse_numerator = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/Veterinary_Numerator.csv", index_col=False)

## only WNV and horse
data_horse_numerator = data_horse_numerator[(data_horse_numerator["Arbovirus"] == "WNV") & (data_horse_numerator["SpeciesName"] == "Equine")]

## convert onsetdate to datetime
data_horse_numerator["onsetdate"] = pd.to_datetime(data_horse_numerator["onsetdate"])

## get Month
data_horse_numerator["Month"] = data_horse_numerator["onsetdate"].dt.month

# drop week, Arbovirus, SpeciesName, onsetdate
data_horse_numerator = data_horse_numerator.drop(["Week", "Arbovirus", "SpeciesName", "onsetdate"], axis=1)

## group by Year, Month, State, County
data_horse_numerator = data_horse_numerator.groupby(['Year', 'Month', 'State', 'County']).size().reset_index(name='Horse_WNV_Count')

## combine disease data and CDC data
# load disease data
data_disease = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
                           "add_0_for_no_wnv/cali_week_wnnd_multi_years_all_features_mosquitoes_bird.csv", index_col=0)

# merge data
data_merge = pd.merge(data_disease, data_horse_numerator, how='left', left_on=['Year', 'Month', 'FIPS'], right_on=['Year', 'Month', 'County'])

# # drop columns that are not features and drop target
data_merge = data_merge.drop(["State_y", "County_y"], axis=1)

# rename the columns
data_merge = data_merge.rename(columns={'State_x': 'State', 'County_x': 'County'})

# fill nan with 0
data_merge["Horse_WNV_Count"] = data_merge["Horse_WNV_Count"].fillna(0)

## output to csv
data_merge.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results"
                            "/add_0_for_no_wnv/cali_week_wnnd_multi_years_all_features_mosquitoes_bird_horse.csv")