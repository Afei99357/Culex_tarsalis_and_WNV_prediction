from sklearn.svm import SVR
from sklearn import metrics, ensemble
import pandas as pd
from sklearn.preprocessing import StandardScaler
import shap
from matplotlib import pyplot as plt
import numpy as np
from hyperopt import fmin, tpe, hp


# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/CA_human_data_2004_to_2023_final_no_impute_0.csv",
                   index_col=False,
                   header=0)

##n choose data only in county: Fresno, Kern, Los Angeles, Merced, Orange, Placer, Riverside, Sacramento, San Bernardino, San Joaquin, Solano, Stanislaus, and Tulare
data = data[data["County"].isin(x.lower() for x in ["Fresno", "Kern", "Los Angeles", "Merced", "Orange", "Placer", "Riverside",
                                 "Sacramento", "San Bernardino", "San Joaquin", "Solano",
                                 "Stanislaus", "Tulare"])]

# Get the Date column to a new dataframe
date = data.pop("Date")

# Drop columns that are not features and drop target
data = data.drop([
    # "Year",
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

# Start to use the earliest first three years to train the model and predict the next year,
# then use the first four years to train the model and predict the next year,
# and so on until the last year
mse_list = []
r2_list = []

# Start predicting 2006
predict_year = 2005
## create two list to store the mean of the predicted values and the mean of the test labels
y_predict_mean_list = []
test_labels_mean_list = []

for year in years:
    if year < predict_year:
        continue
    else:
        train = data[data['Year'] < year].copy()
        test = data[(data['Year'] == year)].copy()

        # Drop rows if they have nan values for both train and test data
        train = train.dropna().reset_index(drop=True)
        test = test.dropna().reset_index(drop=True)

        # Get labels
        train_labels = train.pop("Human_Disease_Count").values
        test_labels = test.pop("Human_Disease_Count").values

        # Remove unnecessary columns
        train.drop(["Month", "FIPS", "Year"], axis=1, inplace=True)
        test.drop(["Month", "FIPS", "Year"], axis=1, inplace=True)

        # Get the column names
        train_column_names = train.columns
        test_column_names = test.columns

        ## tuning the hyperparameters using hyperopt
        def objective(params):
            # HGBR
            hgbr = ensemble.HistGradientBoostingRegressor(**params)
            hgbr.fit(train, train_labels)
            y_predict = hgbr.predict(test)
            ## CALCULATE q2
            q2 = metrics.r2_score(test_labels, y_predict)
            return -q2

        space = {
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.1),
            "max_iter": hp.choice("max_iter", range(50, 200)),
            "max_depth": hp.choice("max_depth", range(1, 50)),
            "max_leaf_nodes": hp.choice("max_leaf_nodes", range(10, 100)),
            "min_samples_leaf": hp.choice("min_samples_leaf", range(1, 10)),
            "l2_regularization": hp.uniform("l2_regularization", 0.0, 1.0)

        }

        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100)
        print(best)