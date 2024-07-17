from sklearn import ensemble, metrics
import pandas as pd
import numpy as np



# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/CA_human_data_2004_to_2023_final_no_impute_0.csv",
                   index_col=False,
                   header=0)

## get the Date column to a new dataframe
date = data.pop("Date")

# drop columns that are not features and drop target
data = data.drop([
    "Year",
    # 'Month',
    "County",
    'FIPS',
    "Latitude",
    "Longitude",
    "Total_Bird_WNV_Count",
    "Mos_WNV_Count",
    "Horse_WNV_Count",
    "average_human_case_monthly",
], axis=1)

## drop the columns if all the values in the columns are the same or all nan
data = data.dropna(axis=1, how='all')

## reindex the data
data = data.reset_index(drop=True)

## print 0 variance columns
print(data.columns[data.var() == 0])

## check if any columns has zero variance and drop the columns
data = data.loc[:, data.var() != 0]

## add the Date column back to the data
data["Date"] = date

# convert "Date" column to datetime
data["Date"] = pd.to_datetime(data["Date"])

# ## only have data from 2006 to 2023
# data = data[data["Date"] >= "2006-01-01"]

## get the unique years and sort them
years = data["Date"].dt.year.unique()
years[::-1].sort()

## drop rows if has nan values
data = data.dropna().reset_index(drop=True)

## start to use the last three years to train the model and predict the previous year,
## then use the last four years to train the model and predict the previous year,
## and so on until the last year
mse_list = []
r2_list = []

## start predicting 2022
predict_year = 2022

for year in years:
    if year > predict_year:
        continue
    else:
        train = data[data['Date'].dt.year > year]
        test = data[(data['Date'].dt.year == year)]

        # Get labels
        train_labels = train.pop("Human_Disease_Count").values
        test_labels = test.pop("Human_Disease_Count").values

        train.pop("Date")
        test.pop("Date")

        # get the column names
        train_column_names = train.columns
        test_column_names = test.columns

        ## Random Forest Classifier
        rf = ensemble.RandomForestRegressor(n_estimators=4, max_depth=3, max_features=0.7, n_jobs=-1)

        rf.fit(train, train_labels)

        y_predict = rf.predict(test)

        mse = metrics.mean_squared_error(test_labels, y_predict)
        r2 = metrics.r2_score(test_labels, y_predict)

        mse_list.append(mse)
        r2_list.append(r2)

        ## print predict year with r2
        print("predict year: ", predict_year, ", Q2: ", r2)

        predict_year -= 1

        ## clear the train and test data
        train = None
        test = None

# print("mse_list: ", mse_list)
# print("r2_list: ", r2_list)


