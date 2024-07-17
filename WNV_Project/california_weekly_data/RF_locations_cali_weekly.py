from sklearn import ensemble, metrics
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
                   "add_0_for_no_wnv/cali_week_wnnd_multi_years_all_features_ebirds.csv", index_col=0)

# drop columns that are not features and drop target
data = data.drop(["State", "County", "Year", 'Month', "County_Seat_Latitude", "County_Seat_Longitude", "FIPS",
                  # "Human_WNND_Count",
                  "Human_WNND_Rate"
                  # "Population"
                  ], axis=1)

# drop the columns that name contains "4m_shift"
# data = data.drop([col for col in data.columns if "4m_shift" in col], axis=1)

data = data.dropna()

# convert "Date" column to datetime
data["Date"] = pd.to_datetime(data["Date"])

# check the standard deviation of the target
print("The standard deviation of the target is: ", data["Human_WNND_Count"].std())

# ## get the "Date" before 2012-05-01 as train data
# train = data[(data["Date"] > "2010-01-01") & (data["Date"] < "2011-01-01")]
# test = data[data["Date"] >= "2011-01-01"]

## split the data into train and test
train, test = train_test_split(data, test_size=0.25, random_state=42)

# Get labels
train_labels = train.pop("Human_WNND_Count").values
test_labels = test.pop("Human_WNND_Count").values

train.pop("Date")
test.pop("Date")

# get the column names
train_column_names = train.columns
test_column_names = test.columns

######################## RF ######################################
# Random Forest Classifier

## grid search for best hyperparameters
rf = ensemble.RandomForestRegressor(n_jobs=-1, random_state=42)

params = {
    'max_depth': [2, 3, 4, 5],
    'min_samples_leaf': [2, 3, 4, 5, 6],
    'n_estimators': [3, 4, 5, 6, 7, 10],
    'max_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
}

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv=4,
                           n_jobs=-1, verbose=1, scoring="neg_mean_absolute_error")

grid_search.fit(train, train_labels)

print(grid_search.best_score_)

rf_best = grid_search.best_estimator_

print(rf_best)

######################### RF ######################################
## Random Forest Classifier

rf_best.fit(train, train_labels)

y_predict = rf_best.predict(test)

# store test labels and predicted labels in a dataframe
df = pd.DataFrame({"test_labels": test_labels, "y_predict": y_predict})
df.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/"
          "results/add_0_for_no_wnv/RF_cali_multi_Years_result.csv")

# ## get the mse, r2 score
mse = metrics.mean_squared_error(test_labels, y_predict)
r2 = metrics.r2_score(test_labels, y_predict)
# for comparison, get the mse and r2 score of a fake model that always predict the mean of the target
fake_model_mse = metrics.mean_squared_error(test_labels, [train_labels.mean()] * len(test_labels))
fake_model_r2 = metrics.r2_score(test_labels, [train_labels.mean()] * len(test_labels))

# #print the mse and r2 score
print("The mean squared error of Random Forest Model verse Fake Model: {:.03}, vs {:.03}".format(mse, fake_model_mse))
print("The r2 score of Random Forest Model verse Fake Model: {:.03}, vs {:.03}".format(r2, fake_model_r2))
####################################################################