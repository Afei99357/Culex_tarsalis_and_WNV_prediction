from sklearn import metrics, ensemble
import pandas as pd
import shap
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, tpe, hp


# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/CA_13_counties_04_23_no_impute.csv",
                   index_col=False,
                   header=0)

# Drop columns that are not features and drop target
data = data.drop([
    "Date",
    "County",
    "Latitude",
    "Longitude",
    "Total_Bird_WNV_Count",
    "Mos_WNV_Count",
    "Horse_WNV_Count",
    # "lai_hv_1m_shift"
], axis=1)

# Drop columns if all the values in the columns are the same or all nan
data = data.dropna(axis=1, how='all')

# Reindex the data
data = data.reset_index(drop=True)

# Print 0 variance columns
print(data.columns[data.var() == 0])

# Check if any columns have zero variance and drop the columns
data = data.loc[:, data.var() != 0]

# Get the unique years and sort them
years = data["Year"].unique()
years.sort()

## impute any missing in Human_Disease_Count with 0
data["Human_Disease_Count"] = data["Human_Disease_Count"].fillna(0)

train = data[data['Year'] < 2019].copy()
test = data[(data['Year'] >= 2019)].copy()

# Drop rows if they have nan values for both train and test data
train = train.dropna().reset_index(drop=True)
test = test.dropna().reset_index(drop=True)

# Get labels
train_labels = train.pop("Human_Disease_Count").values
test_labels = test.pop("Human_Disease_Count").values

# Remove unnecessary columns
train.drop(["Month", "FIPS", "Year"], axis=1, inplace=True)
test.drop(["Month", "FIPS", "Year"], axis=1, inplace=True)

## scale the data
scaler = StandardScaler()
train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
test = pd.DataFrame(scaler.transform(test), columns=test.columns)

# Get the column names
train_column_names = train.columns
test_column_names = test.columns

## tuning the hyperparameters using hyperopt
# Define the mappings for kernel and gamma
scoring_map = ['loss', 'neg_mean_squared_error', 'neg_mean_absolute_error']
def objective(params):
    # int params
    params['max_depth'] = int(params['max_depth'])
    params['max_iter'] = int(params['max_iter'])
    params['max_leaf_nodes'] = int(params['max_leaf_nodes'])
    params['min_samples_leaf'] = int(params['min_samples_leaf'])
    params['max_bins'] = int(params['max_bins'])

    # hgbr
    hgbr = ensemble.HistGradientBoostingRegressor(**params)
    hgbr.fit(train, train_labels)
    y_predict = hgbr.predict(test)
    ## CALCULATE q2
    q2 = metrics.r2_score(test_labels, y_predict)
    return -q2

# Define the hyperparameter space for hgbr
space = {
    'max_depth': hp.quniform('max_depth', 1, 30, 1),
    'max_iter':  hp.quniform('max_iter', 100, 1000, 100),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
    'l2_regularization': hp.uniform('l2_regularization', 0.0, 1.0),
    'max_leaf_nodes': hp.quniform('max_leaf_nodes', 10, 100, 10),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
    'max_bins': hp.quniform('max_bins', 10, 255, 5),
    'scoring': hp.choice('scoring', ['loss', 'neg_mean_squared_error', 'neg_mean_absolute_error'])
}

best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=200)

# Map the best indices back to the corresponding string values
best['scoring'] = scoring_map[int(best['scoring'])]

## int the best hyperparameters
best["max_depth"] = int(best["max_depth"])
best["max_iter"] = int(best["max_iter"])
best["max_leaf_nodes"] = int(best["max_leaf_nodes"])
best["min_samples_leaf"] = int(best["min_samples_leaf"])
best["max_bins"] = int(best["max_bins"])

# Train the model with the best hyperparameters
hgbr = ensemble.HistGradientBoostingRegressor(**best)

hgbr.fit(train, train_labels)

y_predict = hgbr.predict(test)

r2 = metrics.r2_score(test_labels, y_predict)

## RMSE
rmse = np.sqrt(metrics.mean_squared_error(test_labels, y_predict))

print("Q^2: ", r2)
print("RMSE: ", rmse)
print("Best hyperparameters: ", best)