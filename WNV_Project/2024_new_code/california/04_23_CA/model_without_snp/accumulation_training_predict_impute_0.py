from sklearn import ensemble, metrics
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import shap

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/CA_human_data_2004_to_2023_final_no_impute_0.csv",
                   index_col=False,
                   header=0)

# Get the Date column to a new dataframe
date = data.pop("Date")

# Drop columns that are not features and drop target
data = data.drop([
    "Year",
    "County",
    "Latitude",
    "Longitude",
    "Total_Bird_WNV_Count",
    "Mos_WNV_Count",
    "Horse_WNV_Count",
], axis=1)

# Drop columns if all the values in the columns are the same or all nan
data = data.dropna(axis=1, how='all')

# Reindex the data
data = data.reset_index(drop=True)

# Print 0 variance columns
print(data.columns[data.var() == 0])

# Check if any columns have zero variance and drop the columns
data = data.loc[:, data.var() != 0]

# Add the Date column back to the data
data["Date"] = date

# Convert "Date" column to datetime
data["Date"] = pd.to_datetime(data["Date"])

# Get the unique years and sort them
years = data["Date"].dt.year.unique()
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

for year in years:
    if year < predict_year:
        continue
    else:
        train = data[data['Date'].dt.year < year].copy()
        test = data[(data['Date'].dt.year == year)].copy()


        # Drop rows if they have nan values for both train and test data
        train = train.dropna().reset_index(drop=True)
        test = test.dropna().reset_index(drop=True)

        # Get labels
        train_labels = train.pop("Human_Disease_Count").values
        test_labels = test.pop("Human_Disease_Count").values

        # Remove unnecessary columns
        train.drop(["Month", "FIPS", "Date"], axis=1, inplace=True)
        test.drop(["Month", "FIPS", "Date"], axis=1, inplace=True)

        # Get the column names
        train_column_names = train.columns
        test_column_names = test.columns

        # HGBR
        hgbr = ensemble.HistGradientBoostingRegressor(l2_regularization=0.55, learning_rate=0.07, max_depth=28, max_iter=71, max_leaf_nodes=48, min_samples_leaf=7)

        hgbr.fit(train, train_labels)

        y_predict = hgbr.predict(test)

        # ## shap values
        # explainer = shap.TreeExplainer(hgbr)(test)
        #
        # # plt.close()
        # ## plot global bar plot where the global importance of each feature is taken to be the mean absolute value for that feature over all the given samples.
        # plt.figure(figsize=(30, 10))
        #
        # shap.plots.bar(explainer, show=False, max_display=17)
        # plt.tight_layout()
        # plt.savefig(
        #     f"/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/result/plots/shap_summary_accumulated/global_shap_plot_{year}.png")

        mse = metrics.mean_squared_error(test_labels, y_predict)
        r2 = metrics.r2_score(test_labels, y_predict)

        mse_list.append(mse)
        r2_list.append(r2)

        predict_year += 1

        # Print the prediction year with r2
        print("predict year: ", year, ", Q2: ", r2)

        # Clear the train and test data
        train = None
        test = None

print("mse_list: ", mse_list)
print("r2_list: ", r2_list)
