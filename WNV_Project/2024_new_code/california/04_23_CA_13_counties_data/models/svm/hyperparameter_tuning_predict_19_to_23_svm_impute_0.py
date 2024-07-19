from sklearn.svm import SVR
from sklearn import metrics
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

# Define the mappings for kernel and gamma
kernel_map = ["rbf", "linear", "poly", "sigmoid"]
gamma_map = ["scale", "auto"]

## tuning the hyperparameters using hyperopt
def objective(params):
    svm = SVR(**params)
    svm.fit(train, train_labels)
    y_predict = svm.predict(test)
    ## CALCULATE q2
    q2 = metrics.r2_score(test_labels, y_predict)
    return -q2

space = {
    "C": hp.uniform("C", 0.1, 10),
    "epsilon": hp.uniform("epsilon", 0.1, 5),
    "kernel": hp.choice("kernel", ["rbf", "linear", "poly", "sigmoid"]),
    "gamma": hp.choice("gamma", ["scale", "auto"])
}

best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=200)

# Map the best indices back to the corresponding string values
best['kernel'] = kernel_map[int(best['kernel'])]
best['gamma'] = gamma_map[int(best['gamma'])]

# SVM
svm = SVR(**best)

svm.fit(train, train_labels)

y_predict = svm.predict(test)

r2 = metrics.r2_score(test_labels, y_predict)

## RMSE
rmse = np.sqrt(metrics.mean_squared_error(test_labels, y_predict))

print("Q^2: ", r2)
print("RMSE: ", rmse)
print("Best hyperparameters: ", best)