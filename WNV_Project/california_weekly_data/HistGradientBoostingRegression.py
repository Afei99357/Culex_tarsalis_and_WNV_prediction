from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn import ensemble, metrics
import pandas as pd
import numpy as np


# # Load the dataset into a Pandas DataFrame
# data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
#                    "add_0_for_no_wnv/cali_week_wnnd_multi_years_all_features_ebirds.csv", index_col=0)
#
# # drop columns that are not features and drop target
# data = data.drop(["State", "County", "Year", 'Month', "County_Seat_Latitude", "County_Seat_Longitude", "FIPS",
#                   # "Human_WNND_Count",
#                   "Human_WNND_Rate",
#                   # "Population"
#                   ], axis=1)

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
                 "add_0_for_no_wnv/cali_week_wnnd_multi_years_all_features_mosquitoes.csv", index_col=0)

# drop columns that are not features and drop target
data = data.drop(["State",
              "County",
              "Year", 'Month', "County_Seat_Latitude", "County_Seat_Longitude", "FIPS",
                  "Human_WNND_Count",
                  # "Human_WNND_Rate",
                  # "Human_WNF_Count",
                  "Human_WNND_Rate",
                  # "WNV_Mos",
                  # "Population"
              ], axis=1)
# drop row where "WNV_Mos" columns value is NaN
data = data.dropna(subset=["WNV_Mos"])


# convert "Date" column to datetime
data["Date"] = pd.to_datetime(data["Date"])



## get the "Date" before 2012-05-01 as train data
# train = data[(data["Date"] > "2010-01-01") & (data["Date"] < "2012-01-01")]
# test = data[(data["Date"] >= "2012-01-01") & (data["Date"] < "2012-12-31")]

### 111 ####
### train and test data ####
train = data[(data["Date"] < "2011-01-01")]
test = data[(data["Date"] >= "2011-01-01")]

# Get labels
train_labels = train.pop("WNV_Mos").values
test_labels = test.pop("WNV_Mos").values

train.pop("Date")
test.pop("Date")

# get the column names
train_column_names = train.columns
test_column_names = test.columns

######################### RF ######################################
## Random Forest Classifier
est = ensemble.HistGradientBoostingRegressor(max_iter=1000, max_depth=2, max_leaf_nodes=5, learning_rate=0.1)

est.fit(train, train_labels)

y_predict = est.predict(test)

# store test labels and predicted labels in a dataframe
df = pd.DataFrame({"test_labels": test_labels, "y_predict": y_predict})
df.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/"
          "results/add_0_for_no_wnv/EST_cali_multi_Years_result_rate.csv")

# ## get the mse, r2 score
mse = metrics.mean_squared_error(test_labels, y_predict)
r2 = metrics.r2_score(test_labels, y_predict)
# for comparison, get the mse and r2 score of a fake model that always predict the mean of the target
fake_model_mse = metrics.mean_squared_error(test_labels, [train_labels.mean()] * len(test_labels))
fake_model_r2 = metrics.r2_score(test_labels, [train_labels.mean()] * len(test_labels))

# #print the mse and r2 score
# check the standard deviation of the target
print("The standard deviation of the target is: ", data["WNV_Mos"].std())
print("The mean squared error of Histogram-based Gradient Boosting Regression Tree Model verse Fake Model: {:.03}, vs {:.03}".format(mse, fake_model_mse))
print("The r2 score of Histogram-based Gradient Boosting Regression Tree Model verse Fake Model: {:.03}, vs {:.03}".format(r2, fake_model_r2))
