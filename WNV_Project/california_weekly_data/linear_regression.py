## build linear regression model for california weekly west nile virus data

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
                   "add_0_for_no_wnv/cali_week_wnnd_multi_years_all_features_mosquitoes.csv", index_col=0)

# drop columns that are not features and drop target
data = data.drop(["State", "County", "Year", 'Month', "County_Seat_Latitude", "County_Seat_Longitude", "FIPS",
                  "Human_WNND_Count",
                  # "Human_WNND_Rate"
                  "Population"
                  ], axis=1)

data = data.dropna()

# convert "Date" column to datetime
data["Date"] = pd.to_datetime(data["Date"])

# check the standard deviation of the target
print("The standard deviation of the target is: ", data["Human_WNND_Rate"].std())

### train and test data ####
train = data[(data["Date"] > "2011-01-01") & (data["Date"] < "2012-01-01")]
test = data[(data["Date"] >= "2012-01-01") & (data["Date"] < "2012-12-31")]

# Get labels
train_labels = train.pop("Human_WNND_Rate").values
test_labels = test.pop("Human_WNND_Rate").values

train.pop("Date")
test.pop("Date")

# get the column names
train_column_names = train.columns
test_column_names = test.columns

# normalize the data
scaler = StandardScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)

# train the linear regression model
linear_regression = LinearRegression()
linear_regression.fit(train, train_labels)

# get the coefficient and intercept
coefficient = linear_regression.coef_
intercept = linear_regression.intercept_

# get the score of the model
score = linear_regression.score(train, train_labels)

# get the prediction of the model
prediction = linear_regression.predict(test)

# get the mse
mse = metrics.mean_squared_error(test_labels, prediction)

# get the r2 score
r2 = metrics.r2_score(test_labels, prediction)

# for comparison, get the mse and r2 score of a fake model that always predict the mean of the target
fake_model_mse = metrics.mean_squared_error(test_labels, [train_labels.mean()] * len(test_labels))
fake_model_r2 = metrics.r2_score(test_labels, [train_labels.mean()] * len(test_labels))

# #print the mse and r2 score
print("The mean squared error of Linear Regression Model verse Fake Model: {:.03}, vs {:.03}".format(mse, fake_model_mse))
print("The r2 score of Linear Regression Model verse Fake Model: {:.03}, vs {:.03}".format(r2, fake_model_r2))



