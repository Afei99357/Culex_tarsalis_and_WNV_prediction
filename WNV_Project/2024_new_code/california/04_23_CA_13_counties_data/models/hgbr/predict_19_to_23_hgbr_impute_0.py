from sklearn import ensemble, metrics
import pandas as pd
import shap
from matplotlib import pyplot as plt
import numpy as np

# Load the dataset into a Pandas DataFrame
data = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/CA_13_counties_04_23_no_impute_daylight.csv",
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

## training data before 2019
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

test_FIPS_list = test.pop("FIPS").values
test_month_list = test.pop("Month").values
test_year_list = test.pop("Year").values

# Get the column names
train_column_names = train.columns
test_column_names = test.columns

## read the best hyperparameter from the file
best_hyperparameter = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/"
                                  "CA_13_county_dataset/result/plots/HGBR/hyperparameter_tuning_plots/hyperparameter_tuning_hgbr_impute_0.csv",
                                  header=0, index_col=False)

## create a dataframe to store the Q^2 and RMSE, and tuning year and hyperparameters
df_output = pd.DataFrame(columns=["tuning_year", "Q^2", "RMSE", "max_depth", "max_iter", "learning_rate",
                                    "l2_regularization", "max_leaf_nodes", "min_samples_leaf", "max_bins", "scoring"])

## create a list to store the all the Q^2 and RMSE and tuning year and hyperparameters
output_list = []

for index, row in best_hyperparameter.iterrows():
    hyperparameter_tuning_year = row['tuning_year']
    max_depth = int(row["max_depth"])
    max_iter = int(row["max_iter"])
    learning_rate = float(row["learning_rate"])
    l2_regularization = float(row["l2_regularization"])
    max_leaf_nodes = int(row["max_leaf_nodes"])
    min_samples_leaf = int(row["min_samples_leaf"])
    max_bins = int(row["max_bins"])
    scoring = row["scoring"]

    model = ensemble.HistGradientBoostingRegressor(max_depth=max_depth,
                                                   max_iter=max_iter,
                                                   learning_rate=learning_rate,
                                                   l2_regularization=l2_regularization,
                                                   max_leaf_nodes=max_leaf_nodes,
                                                   min_samples_leaf=min_samples_leaf,
                                                   max_bins=max_bins,
                                                   scoring=scoring)

    model.fit(train, train_labels)

    # Predict the test data
    predictions = model.predict(test)

    # Calculate the Q^2 and RMSE
    q2 = metrics.r2_score(test_labels, predictions)

    # Calculate the RMSE
    rmse = np.sqrt(metrics.mean_squared_error(test_labels, predictions))

    output_list.append([hyperparameter_tuning_year, q2, rmse, max_depth, max_iter, learning_rate, l2_regularization,
                        max_leaf_nodes, min_samples_leaf, max_bins, scoring])

    ## print the Q^2 and RMSE with tuning year
    print(f"tuning year: {hyperparameter_tuning_year}, Q^2: {q2}, RMSE: {rmse}")

## add the output list to the dataframe
df_output = pd.DataFrame(output_list, columns=["tuning_year", "q2", "RMSE", "max_depth", "max_iter", "learning_rate",
                                                  "l2_regularization", "max_leaf_nodes", "min_samples_leaf", "max_bins",
                                                    "scoring"])

## save the tuning year, q2 and RMSE to a csv file
df_output.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/result/"
                 "plots/hgbr/hgbr_impute_0_tuning_year_q2_rmse.csv", index=False)