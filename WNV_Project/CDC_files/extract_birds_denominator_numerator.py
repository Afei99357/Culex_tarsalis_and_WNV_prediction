import numpy as np
import pandas as pd

# Load the dataset into a Pandas DataFrame
data_bird_numerator = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/Dead_Bird_Numerator.csv",
                       index_col=False)

data_bird_denominator = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/Dead_Bird_Denominator.csv",
                          index_col=False)

# combine Year and Week into one column
data_bird_numerator["Date"] = data_bird_numerator["Year"].astype(str) + "_" + data_bird_numerator["Week"].astype(str)
data_bird_denominator["Date"] = data_bird_denominator["Year"].astype(str) + "_" + data_bird_denominator["Week"].astype(str)

# based on Year and Week, get the date of the first day of the week
data_bird_numerator["Date"] = pd.to_datetime(data_bird_numerator["Date"] + '-1', format='%Y_%W-%w')
data_bird_denominator["Date"] = pd.to_datetime(data_bird_denominator["Date"] + '-1', format='%Y_%W-%w')

# get the month and year
data_bird_numerator["Month"] = data_bird_numerator["Date"].dt.month
data_bird_denominator["Month"] = data_bird_denominator["Date"].dt.month

# only keep the data with Arbovirus is WNV
data_bird_numerator = data_bird_numerator[data_bird_numerator["Arbovirus"] == "WNV"]
data_bird_denominator = data_bird_denominator[data_bird_denominator["Arbovirus"] == "WNV"]

# drop Week, Arbovirus and Date columns
data_bird_denominator.drop(["Week", "Arbovirus", "Date"], axis=1, inplace=True)
data_bird_numerator.drop(["Week", "Arbovirus", "Date"], axis=1, inplace=True)

# get Corvid and non corvid disease data
data_bird_numerator_corvid = data_bird_numerator[data_bird_numerator["Corvid"] == 1]
data_bird_numerator_non_cor = data_bird_numerator[data_bird_numerator["Corvid"] == 0]

# group by Year, Month, State and County and sum up the number of positive as number of birds
data_bird_denominator = data_bird_denominator.groupby(["Year", "Month", "State", "County"]).sum()

# reset index
data_bird_denominator.reset_index(inplace=True)

# group by Year, Month, State and County and  the number of positive corvid as WNV_Corvid_Count
data_bird_numerator_corvid = data_bird_numerator_corvid.groupby(["Year", "Month", "State", "County"]).size().reset_index(name='WNV_Corvid_Count')

# group by Year, Month, State and County and sum up the number of positive non corvid as WNV_NonCorvid_Count
data_bird_numerator_non_cor = data_bird_numerator_non_cor.groupby(["Year", "Month", "State", "County"]).size().reset_index(name='WNV_NonCorvid_Count')

# merge the corvid and non corvid data to data_bird_denominator
data_merge = pd.merge(data_bird_denominator, data_bird_numerator_corvid, how='outer', on=["Year", "Month", "State", "County"])

data_merge = pd.merge(data_merge, data_bird_numerator_non_cor, how='outer', on=["Year", "Month", "State", "County"])

# as long as CorvidsTested is not null and gretaer than 0, if WNV_Corvid_Count is null, set it to 0
data_merge.loc[(data_merge['CorvidsTested'].notnull())
               & (data_merge['CorvidsTested'] > 0)
               & (data_merge['WNV_Corvid_Count'].isnull()), 'WNV_Corvid_Count'] = 0

# as long as OtherTested is not null and gretaer than 0, if WNV_NonCorvid_Count is null, set it to 0
data_merge.loc[(data_merge['OtherTested'].notnull())
               & (data_merge['OtherTested'] > 0)
               & (data_merge['WNV_NonCorvid_Count'].isnull()), 'WNV_NonCorvid_Count'] = 0


# make a copy of column WNV_Corvid_Count and add WNV_NonCorvid_Count, name them corvids and other
data_merge["Corvid_Count"] = data_merge["WNV_Corvid_Count"]
data_merge["NonCorvid_Count"] = data_merge["WNV_NonCorvid_Count"]

# fill nan with 0
data_merge["Corvid_Count"] = data_merge["Corvid_Count"].fillna(0)
data_merge["NonCorvid_Count"] = data_merge["NonCorvid_Count"].fillna(0)

# add a column total_bird_WNV_Count by adding up the WNV_Count of Corvids and Other
data_merge["Total_Bird_WNV_Count"] = data_merge["Corvid_Count"] + data_merge["NonCorvid_Count"]

# if both WNV_Corvid_Count and WNV_NonCorvid_Count are Nan, then replace total_bird_WNV_Count with Nan
data_merge.loc[data_merge["WNV_NonCorvid_Count"].isna() & data_merge["WNV_Corvid_Count"].isna(), "Total_Bird_WNV_Count"] = None


# drop the columns Corvid_Count and NonCorvid_Count
data_merge = data_merge.drop(["Corvid_Count", "NonCorvid_Count"], axis=1)

# output the data to csv file
data_merge.to_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/Dead_Bird_Denominator_Numerator_monthly.csv")

