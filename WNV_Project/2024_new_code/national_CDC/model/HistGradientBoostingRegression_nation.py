from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn import ensemble, metrics
import pandas as pd
import numpy as np


# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/"
                   "human/cdc_human_1999_to_2023/WNV_human_and_non_human_yearly_climate_demographic_bird.csv", index_col=0)

## remove any space and comma in Population column
data["Population"] = data["Population"].str.replace(",", "").str.strip()

## convert Population column to numeric
data["Population"] = pd.to_numeric(data["Population"], errors='coerce')

# select the columns after column Date as predictors
date_index = data.columns.get_loc("Date")

# Select columns after the "Date" column as predictors
data_pred = data.iloc[:, date_index+1:]

# get Year column from data and add it to data_pred
data_pred['Year'] = data['Year']

## add target column
data_pred["Neuroinvasive_disease_cases"] = data["Neuroinvasive_disease_cases"]

# drop nan values
data_pred = data_pred.dropna()

## reset index
data_pred = data_pred.reset_index(drop=True)

### train and test data ####
train = data_pred[(data_pred["Year"] < 2018)]
test = data_pred[(data_pred["Year"] >= 2018)]

# Get labels
train_labels = train.pop("Neuroinvasive_disease_cases").values
test_labels = test.pop("Neuroinvasive_disease_cases").values

## remove time column
train.pop("Year")
test.pop("Year")

# get the column names
train_column_names = train.columns
test_column_names = test.columns

######################### HistGradientBoostingRegressor ######################################
## HistGradientBoostingRegressor
est = ensemble.HistGradientBoostingRegressor(max_iter=1000, max_depth=2, max_leaf_nodes=5, learning_rate=0.1)

est.fit(train, train_labels)

y_predict = est.predict(test)

# store test labels and predicted labels in a dataframe
df = pd.DataFrame({"test_labels": test_labels, "y_predict": y_predict})
df.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/Histogram-Based Gradient Boosting Ensembles/EST_cali_multi_Years_result.csv")

# ## get the mse, r2 score
mse = metrics.mean_squared_error(test_labels, y_predict)
r2 = metrics.r2_score(test_labels, y_predict)
# for comparison, get the mse and r2 score of a fake model that always predict the mean of the target
fake_model_mse = metrics.mean_squared_error(test_labels, [train_labels.mean()] * len(test_labels))
fake_model_r2 = metrics.r2_score(test_labels, [train_labels.mean()] * len(test_labels))

# #print the mse and r2 score
# check the standard deviation of the target
print("The standard deviation of the target is: ", data["Neuroinvasive_disease_cases"].std())
print("The mean squared error of Histogram-based Gradient Boosting Regression Tree Model verse Fake Model: {:.03}, vs {:.03}".format(mse, fake_model_mse))
print("The r2 score of Histogram-based Gradient Boosting Regression Tree Model verse Fake Model: {:.03}, vs {:.03}".format(r2, fake_model_r2))
