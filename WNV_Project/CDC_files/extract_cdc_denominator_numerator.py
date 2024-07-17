import pandas as pd

# # Load the dataset into a Pandas DataFrame
data_denominator = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/Mosquito_Denominator.csv", index_col=False)

# choose only Arbovirus is WNV
data_denominator = data_denominator[data_denominator["Arbovirus"] == "WNV"]

# combine Year and Week into one column
data_denominator["Year_Week"] = data_denominator["Year"].astype(str) + "_" + data_denominator["Week"].astype(str)

# based on Year and Week, get the date of the first day of the week
data_denominator["Year_Week"] = pd.to_datetime(data_denominator["Year_Week"] + '-1', format='%Y_%W-%w')

# get the month and year
data_denominator["Month"] = data_denominator["Year_Week"].dt.month

# drop Arbovirus, name and Year_Week columns
data_denominator.pop("Arbovirus")
data_denominator.pop("name")
data_denominator.pop('Week')
data_denominator.pop("Year_Week")

# sum up Collected and Tested columns separately based on the Year, Month, State and County
data_denominator = data_denominator.groupby(["Year", "Month", "State", "County"]).sum()

# separate index into Year, Month, State and County
data_denominator = data_denominator.reset_index()

# save the data
# data.to_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/Mosquito_Denominator_WNV_monthly.csv")

## read data numerator file
data_numerator = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/Mosquito_Numerator.csv", index_col=False)

# choose only Arbovirus is WNV
data_numerator = data_numerator[data_numerator["Arbovirus"] == "WNV"]

# get month from datecollected
data_numerator["datecollected"] = pd.to_datetime(data_numerator["datecollected"])

# get the month and year
data_numerator["Month"] = data_numerator["datecollected"].dt.month

# drop Arbovirus, name and datecollected columns
data_numerator.pop("Arbovirus")
data_numerator.pop("name")
data_numerator.pop("datecollected")
data_numerator.pop("Week")

# group by Year, Month, State and County and sum up the number of positive as numer of WNV
data_numerator = data_numerator.groupby(["Year", "Month", "State", "County"]).size().reset_index(name='WNV_Count')

# merge the two dataframes
data_merge = pd.merge(data_denominator, data_numerator, on=["Year", "Month", "State", "County"], how="outer", suffixes=("_denominator", "_numerator"))

## if the Tested column is not 0 or NaN, then fill in the WNV_Count column with 0
for index, row in data_merge.iterrows():
    if row["Tested"] == 0 or pd.isnull(row["Tested"]):
        continue
    else:
        data_merge.loc[index, "WNV_Count"] = 0

data_merge.to_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/Mosquito_Denominator_Numerator_WNV_monthly.csv")
