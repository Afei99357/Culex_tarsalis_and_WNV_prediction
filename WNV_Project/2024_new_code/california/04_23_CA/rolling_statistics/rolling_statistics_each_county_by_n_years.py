import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/CA_human_data_2004_to_2023_final_no_impute_0.csv",
                   index_col=False,
                   header=0)

## get the Date column to a new dataframe
date = data.pop("Date")
county = data.pop("County")

# drop columns that are not features and drop target
data = data.drop([
    "Year",
    # 'Month',
    # "County",
    'FIPS',
    "Latitude",
    "Longitude",
    "Total_Bird_WNV_Count",
    "Mos_WNV_Count",
    "Horse_WNV_Count",
], axis=1)

## drop the columns if all the values in the columns are the same or all nan
data = data.dropna(axis=1, how='all')

## reindex the data
data = data.reset_index(drop=True)

## print 0 variance columns
print(data.columns[data.var() == 0])

## check if any columns has zero variance and drop the columns
data = data.loc[:, data.var() != 0]

## add the Date and county column back to the data
data["Date"] = date
data["County"] = county

# convert "Date" column to datetime
data["Date"] = pd.to_datetime(data["Date"])

### train and test data ####
train = data[data['Date'] < '2021-01-01']
## get the test data in 2020
test = data[data['Date'] >= '2021-01-01']

## impute the nan values with the mean of the column
# ## adding an empty column average_cases column to both train and test data
train["average_human_case_monthly"] = ""

# ## find unique years in train data
years = train["Date"].dt.year.unique()

## calculate the average human cases monthly for each FIPS and month using train
train["average_human_case_monthly"] = train.groupby(["County", "Month"])["Human_Disease_Count"].transform("sum") / len(years)

## based on the FIPS and month, filling average_human_case_monthly from train to test
test = pd.merge(test, train[["County", "Month", "average_human_case_monthly"]].drop_duplicates(), on=["County", "Month"], how="left")

## use the train dataset to impute 0
train.loc[train["Human_Disease_Count"].isna(), "Human_Disease_Count"] = train["average_human_case_monthly"]
test.loc[test["Human_Disease_Count"].isna(), "Human_Disease_Count"] = test["average_human_case_monthly"]

## drop rows if has nan values for both train and test data
train = train.dropna().reset_index(drop=True)
test = test.dropna().reset_index(drop=True)

## Adding Year and Month columns to the data
train["Year"] = train["Date"].dt.year

## get rolling statistics for the train data by every K years for each county
rolling_window = 3
## group train by FIPS and Year and get the rolling statistics
train_roll = train.groupby(["County", "Year"]).agg({"Human_Disease_Count": "sum"}).reset_index()

## get the rolling statistics for each county
train_roll["rolling_mean"] = train_roll["Human_Disease_Count"].rolling(rolling_window).mean()
train_roll["rolling_std"] = train_roll["Human_Disease_Count"].rolling(rolling_window).std()

## plot the rolling statistics for each county
## get unique FIPS
Counties = train_roll["County"].unique()



## rainbow coolor
for i, county in zip(range(len(Counties)), Counties):
    plt.figure(figsize=(14, 7))
    plt.plot(train_roll[train_roll["County"] == county]["Year"], train_roll[train_roll["County"] == county]["Human_Disease_Count"], label='Actual Cases', color="black")
    plt.plot(train_roll[train_roll["County"] == county]["Year"], train_roll[train_roll["County"] == county]["rolling_mean"], label='Rolling Mean', linestyle='--', color="red")
    plt.plot(train_roll[train_roll["County"] == county]["Year"], train_roll[train_roll["County"] == county]["rolling_std"], label='Rolling Std Dev', linestyle=':', color="blue")
    plt.xlabel('Year')
    plt.ylabel('WNV Cases')
    plt.title(f'WNV Cases with Rolling Statistics for County: {county}')
    ## keep x axis to be integer
    plt.xticks(np.arange(2004, 2024, 1), rotation=45, fontsize=5)

    plt.legend(loc='upper right', fontsize=3.6)
    plt.savefig(f'/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/result/plots/rolling_statistics/rolling_statistics_county_{county}.png', dpi=300)