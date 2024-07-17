from sklearn import ensemble, metrics
import pandas as pd
from sklearn.model_selection import GridSearchCV

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/Data/"
                   "human_neuroinvasive_wnv_ebirds.csv", index_col=0)

### southern_california_counties = ["Los Angeles", "San Diego", "Orange", "Riverside", "San Bernardino", "Kern", "Ventura","Santa Barbara", "San Luis Obispo", "Imperial"]

### in FIPS
southern_california_counties = [6037, 6073, 6059, 6065, 6071, 6029, 6111, 6083, 6079, 6025]

data = data[data["FIPS"].isin(southern_california_counties) | data["State"].isin(['North Dakota', 'South Dakota', 'Colorado'])]

# data = data[data["State"].isin(['California', 'North Dakota', 'South Dakota', 'Colorado'])]
data = data[data["FIPS"].isin(southern_california_counties)]
# # drop columns that are not features and drop target
data = data.drop(["FIPS", "County", "State", "State_Code", 'SET', "County_Seat", "County_Seat_Latitude",
                  "County_Seat_Longitude", "County_Centroid_Latitude", "County_Centroid_Longitude",
                  'Poverty_Estimate_All_Ages',
                  # "Population",
                  "State_Land_Area", "Land_Change_Count_Since_1992",
                  "Land_Use_Class", "Processed_Flag_Land_Use", "WNV_Rate_Neural_With_All_Years",
                  # "WNV_Rate_Neural_Without_99_21",
                  "WNV_Rate_Non_Neural_Without_99_21",
                  "State_Horse_WNV_Rate", "WNV_Rate_Non_Neural_Without_99_21_log",
                  "WNV_Rate_Neural_Without_99_21_log"# target column
                  ], axis=1)

### drop monthly weather data block #######################
## get the column u10_Jan and column swvl1_Dec index
column_Poverty_index = data.columns.get_loc("Poverty_Rate_Estimate_All_Ages")
column_u10_Jan_index = data.columns.get_loc("u10_Jan")
column_swvl1_Dec_index = data.columns.get_loc("swvl1_Dec")
column_tp_acc_extrem_index = data.columns.get_loc("tp_acc_Oct_to_Aug")

## DROP the columns between column_u10_Jan and column_swvl1_Dec includes column_u10_Jan and column_swvl1_Dec
# data = data.drop(data.columns[column_u10_Jan_index:column_swvl1_Dec_index + 1], axis=1)
data = data.drop(data.columns[column_u10_Jan_index:column_swvl1_Dec_index + 1], axis=1)
################################################################

data = data.dropna()

train = data[(data["Year"] < 2020) & (data["Year"] >= 2003)]
test = data[(data["Year"] >= 2020) & (data["Year"] < 2022)]

# Get labels
train_labels = train.pop("WNV_Rate_Neural_Without_99_21").values
test_labels = test.pop("WNV_Rate_Neural_Without_99_21").values

train.pop("Year")
test.pop("Year")

train_population = train.pop("Population").values
test_population = test.pop("Population").values

# get the column names
train_column_names = train.columns
test_column_names = test.columns
######################### RF ######################################
## Random Forest Classifier
#
# ## grid search for best hyperparameters
# rf = ensemble.RandomForestRegressor(n_jobs=-1, random_state=42)
#
# params = {
#     'max_depth': [2, 3, 5, 10, 20],
#     'min_samples_leaf': [5, 10, 20, 50, 100, 200],
#     'n_estimators': [10, 25, 30, 50, 100, 200]
# }
#
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator=rf,
#                            param_grid=params,
#                            cv=4,
#                            n_jobs=-1, verbose=1, scoring="neg_mean_absolute_error")
#
# grid_search.fit(train, train_labels)
#
# print(grid_search.best_score_)
#
# rf_best = grid_search.best_estimator_
#
# print(rf_best)

# train the model
rf = ensemble.RandomForestRegressor(n_estimators=10, max_depth=2, min_samples_leaf=5, n_jobs=-1, random_state=42)

rf.fit(train, train_labels)

y_predict = rf.predict(test)

# ## get the mse
mse = metrics.mean_squared_error(test_labels, y_predict)
r2 = metrics.r2_score(test_labels, y_predict)

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