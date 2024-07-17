from sklearn import ensemble, metrics
import pandas as pd
import numpy as np

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/CA_human_data_2004_to_2023_final_no_impute_0.csv",
                   index_col=False,
                   header=0)

##n choose data only in county: Fresno, Kern, Los Angeles, Merced, Orange, Placer, Riverside, Sacramento, San Bernardino, San Joaquin, Solano, Stanislaus, and Tulare
data = data[data["County"].isin(x.lower() for x in ["Fresno", "Kern", "Los Angeles", "Merced", "Orange", "Placer", "Riverside",
                                 "Sacramento", "San Bernardino", "San Joaquin", "Solano",
                                 "Stanislaus", "Tulare"])]

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
    "Horse_WNV_Count",
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

# Start to use the earliest first three years to train the model and predict the next year,
# then use the first four years to train the model and predict the next year,
# and so on until the last year
mse_list = []
r2_list = []

# Start predicting 2006
predict_year = 2005

for year in years:
    if year < predict_year:
        continue
    else:
        train = data[data['Date'].dt.year < year].copy()
        test = data[(data['Date'].dt.year == year)].copy()

        # Impute the nan values with the mean of the column
        # Adding an empty column average_human_case_monthly to both train and test data
        # train["average_human_case_monthly"] = np.nan

        # Find unique years in train data
        train_years = train["Date"].dt.year.unique()

        # # Calculate the average human cases monthly for each FIPS and month using train
        # train["average_human_case_monthly"] = train.groupby(["FIPS", "Month"])["Human_Disease_Count"].transform(
        #     lambda x: x.sum() / len(train_years))
        #
        # # Use the train dataset to impute 0
        # train.loc[train["Human_Disease_Count"].isna(), "Human_Disease_Count"] = train["average_human_case_monthly"]

        ## impute the nan values with 0 for booth train and test
        train.loc[train["Human_Disease_Count"].isna(), "Human_Disease_Count"] = 0
        test.loc[test["Human_Disease_Count"].isna(), "Human_Disease_Count"] = 0

        ## print number of test data and
        # print("year: ", year, ", number of test data: ", len(test))

        # Drop rows if they have nan values for both train and test data
        train = train.dropna().reset_index(drop=True)
        test = test.dropna().reset_index(drop=True)

        ## print number of test data and
        # print("year: ", year, ", number of test data after dropna: ", len(test))

        # Get labels
        train_labels = train.pop("Human_Disease_Count").values
        test_labels = test.pop("Human_Disease_Count").values

        # Remove unnecessary columns
        train.drop(["Month", "FIPS", "Date"], axis=1, inplace=True)
        test.drop(["Month", "FIPS", "Date"], axis=1, inplace=True)

        # Get the column names
        train_column_names = train.columns
        test_column_names = test.columns

        # HGBR
        hgbr = ensemble.HistGradientBoostingRegressor(max_depth=20)

        hgbr.fit(train, train_labels)

        y_predict = hgbr.predict(test)

        mse = metrics.mean_squared_error(test_labels, y_predict)
        r2 = metrics.r2_score(test_labels, y_predict)

        mse_list.append(mse)
        r2_list.append(r2)

        predict_year += 1

        # Print the prediction year with r2
        print("predict year: ", year, ", Q2: ", r2)

        # Clear the train and test data
        train = None
        test = None

print("mse_list: ", mse_list)
print("r2_list: ", r2_list)
