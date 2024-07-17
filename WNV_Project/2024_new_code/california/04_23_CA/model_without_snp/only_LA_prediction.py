from sklearn import ensemble, metrics
import pandas as pd
import numpy as np

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/CA_human_data_2004_to_2023_final_no_impute_0.csv",
                   index_col=False,
                   header=0)

## only choose los angeles county

county = "fresno"

data = data[data["County"] == county]

## reindex the data
data = data.reset_index(drop=True)

# Get the Date column to a new dataframe
date = data.pop("Date")

# Drop columns that are not features and drop target
data = data.drop([
    "Year",
    "County",
    "Latitude",
    "Longitude",
    "Total_Bird_WNV_Count",
    "Mos_WNV_Count",
    "Horse_WNV_Count"
], axis=1)

# Drop columns if all the values in the columns are the same or all nan
data = data.dropna(axis=1, how='all')

# Reindex the data
data = data.reset_index(drop=True)

# Print 0 variance columns
print(data.columns[data.var() == 0])

# Check if any columns have zero variance and drop the columns
data = data.loc[:, data.var() != 0]

# Add the Date column back to the data
data["Date"] = date

# Convert "Date" column to datetime
data["Date"] = pd.to_datetime(data["Date"])

# Get the unique years and sort them
years = data["Date"].dt.year.unique()
years.sort()

# Start from the earliest year, choose 5 years as training data and the next 1 year as test data. Record all the mse and r2 score. Doing this for all the years
mse_list = []
r2_list = []

train_years = 1

for i in range(len(years) - train_years):
    train = data[data['Date'] < str(years[i + train_years]) + '-01-01'].copy()
    test = data[(data['Date'] >= str(years[i + train_years]) + '-01-01') & (data['Date'] < str(years[i + train_years ]) + '-12-31')].copy()

    # Impute the nan values with the mean of the column
    # Adding an empty column average_human_case_monthly to both train and test data
    train["average_human_case_monthly"] = np.nan

    # Calculate the average human cases monthly for each month using train
    train["average_human_case_monthly"] = train.groupby(["Month"])["Human_Disease_Count"].transform(
        lambda x: x.sum() / len(years))

    # Use the train dataset to impute 0
    train.loc[train["Human_Disease_Count"].isna(), "Human_Disease_Count"] = train["average_human_case_monthly"]

    # Drop rows if they have nan values for both train and test data
    train = train.dropna().reset_index(drop=True)
    test = test.dropna().reset_index(drop=True)

    # Get labels
    train_labels = train.pop("Human_Disease_Count").values
    test_labels = test.pop("Human_Disease_Count").values

    # Remove unnecessary columns
    train.drop(["Month", "Date", "average_human_case_monthly"], axis=1, inplace=True)
    test.drop(["Month", "Date"], axis=1, inplace=True)

    # Get the column names
    train_column_names = train.columns
    test_column_names = test.columns

    # HistGradientBoostingRegressor
    HGBR = ensemble.HistGradientBoostingRegressor(max_depth=50)

    HGBR.fit(train, train_labels)

    y_predict = HGBR.predict(test)

    # Store test labels and predicted labels in a dataframe
    df = pd.DataFrame({"test_labels": test_labels, "y_predict": y_predict})
    df.to_csv(f"/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/result/Histogram-Based Gradient Boosting Ensembles/EST_{county}_03_to_24_result.csv")

    # Get the mse, r2 score
    mse = metrics.mean_squared_error(test_labels, y_predict)
    r2 = metrics.r2_score(test_labels, y_predict)

    # For comparison, get the mse and r2 score of a fake model that always predicts the mean of the target
    fake_model_mse = metrics.mean_squared_error(test_labels, [train_labels.mean()] * len(test_labels))
    fake_model_r2 = metrics.r2_score(test_labels, [train_labels.mean()] * len(test_labels))

    mse_list.append(mse)
    r2_list.append(r2)

    # Print Q2 score
    print("predict year: ", years[i + train_years], ", Q2: ", r2)

print("mse_list: ", mse_list)
print("q2_list: ", r2_list)
