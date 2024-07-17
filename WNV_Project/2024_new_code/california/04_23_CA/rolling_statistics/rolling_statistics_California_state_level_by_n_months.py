import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/CA_human_data_2004_to_2023_final_no_impute_0.csv",
                   index_col=False,
                   header=0)

## drop Date
data.pop("Date")

## build up a new Date column with year and month
data["Date"] = pd.to_datetime(data["Year"].astype(str) + "-" + data["Month"].astype(str))

# Extract and reinsert Date and County columns
date = data.pop("Date")
county = data.pop("County")

# Drop columns not needed for analysis
data = data.drop([
    "Year", "FIPS", "Latitude", "Longitude",
    "Total_Bird_WNV_Count", "Mos_WNV_Count", "Horse_WNV_Count",
], axis=1)

# Drop columns with all NaN or zero variance
data = data.dropna(axis=1, how='all')
data = data.loc[:, data.var() != 0]

# Reinsert Date and County columns
data["Date"] = date
data["County"] = county

# Convert "Date" column to datetime with year and month

# Train-test split
train = data[data['Date'] < '2024-01']
test = data[data['Date'] >= '2021-01']

# Impute NaN values with mean
train["average_human_case_monthly"] = train.groupby(["County", "Month"])["Human_Disease_Count"].transform("mean")

# Merge average human cases from train to test based on County and Month
test = pd.merge(test, train[["County", "Month", "average_human_case_monthly"]].drop_duplicates(), on=["County", "Month"], how="left")

# Impute missing values in Human_Disease_Count with average_human_case_monthly
train["Human_Disease_Count"].fillna(train["average_human_case_monthly"], inplace=True)
test["Human_Disease_Count"].fillna(test["average_human_case_monthly"], inplace=True)

# Drop rows with remaining NaN values
train = train.dropna().reset_index(drop=True)
test = test.dropna().reset_index(drop=True)

# Add Year column to train
train["Year"] = train["Date"].dt.year

# Calculate rolling statistics with a 3-year window
rolling_window = 3
train_roll = train.groupby("Date")["Human_Disease_Count"].sum().reset_index()
train_roll["rolling_mean"] = train_roll["Human_Disease_Count"].rolling(window=rolling_window, min_periods=1).mean()
train_roll["rolling_std"] = train_roll["Human_Disease_Count"].rolling(window=rolling_window, min_periods=1).std()


# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(train_roll["Date"], train_roll["Human_Disease_Count"], label='Actual Cases', color="black")
plt.plot(train_roll["Date"], train_roll["rolling_mean"], label='Rolling Mean', linestyle='--', color="red")
plt.plot(train_roll["Date"], train_roll["rolling_std"], label='Rolling Std Dev', linestyle=':', color="blue")
# plt.plot(train_roll["Year"], train_roll["rolling_var"], label='Rolling Variance', linestyle='-.', color="green")
plt.xlabel('Date')
plt.ylabel('WNV Cases')
plt.title(f'WNV Cases with Rolling Statistics for California with {rolling_window}-months Window')

## make sure X axis fropm 2004-01 to 2023-12
plt.xlim(pd.Timestamp('2004-01-01'), pd.Timestamp('2023-12-01'))

plt.legend(loc='upper right')
plt.savefig(f'/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/result/plots/rolling_statistics/california_04_23_{rolling_window}_months_rolling_statistics.png', dpi=300)
