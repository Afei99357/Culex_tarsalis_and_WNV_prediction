from sklearn.svm import SVR
from sklearn import metrics
import pandas as pd
import shap
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, tpe, hp
import seaborn as sns


# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/"
                   "CA_13_counties_04_23_no_impute_daylight.csv",
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

# Start predicting 2006
tuning_years_list = [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]

## create a dictionary to store key as the tuning year and value as another dictionary to store the q2 and mse for each accumulated year using the best hypoerparameters from the tuning year
tuning_year_dict = {}

## store the best hyperparameters in a dataframe
best_hyperparameters_df = pd.DataFrame(columns=["tuning_year", "C", "epsilon", "kernel", "gamma"])

for tuning_year in tuning_years_list:
    ## save the best hyperparameters
    best_hyperparameters = []
    for year in years:
        if year != tuning_year:
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

            ## update the best hyperparameters
            best_hyperparameters.append(best)

            ## store the best hyperparameters in a dataframe
            best_hyperparameters_df = best_hyperparameters_df.append({"tuning_year": year, "C": best["C"], "epsilon": best["epsilon"], "kernel": best["kernel"], "gamma": best["gamma"]}, ignore_index=True)

    print("best_hyperparameters: ", best_hyperparameters)
    # Start to use the earliest first three years to train the model and predict the next year,
    # then use the first four years to train the model and predict the next year,
    # and so on until the last year
    rmse_list = []
    r2_list = []

    # Start predicting 2006
    predict_year = 2005

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

            ## scale the data
            scaler = StandardScaler()
            train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
            test = pd.DataFrame(scaler.transform(test), columns=test.columns)

            # Get the column names
            train_column_names = train.columns
            test_column_names = test.columns

            # SVM
            svm = SVR(**best_hyperparameters[0])
            svm.fit(train, train_labels)

            y_predict = svm.predict(test)

            mse = metrics.mean_squared_error(test_labels, y_predict)
            #RMSE
            rmse = np.sqrt(mse)
            r2 = metrics.r2_score(test_labels, y_predict)


            rmse_list.append(rmse)
            r2_list.append(r2)

            predict_year += 1

            # Print the prediction year with r2
            print("predict year: ", year, ", Q2: ", r2)

            # Clear the train and test data
            train = None
            test = None

    #   ## store the q2 and mse for each accumulated year using the best hyperparameters from the tuning year
    tuning_year_dict[tuning_year] = {"q2": r2_list, "rmse": rmse_list}

## save the best hyperparameters in a csv file
best_hyperparameters_df.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/result/plots/hyperparameter_tuning_plots/hyperparameter_tuning_svm_impute_0_with_daylight_best_hyperparameters.csv", index=False)

x_axis_years = [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

## plot heatmap where the x-axis is the x_axis_years and the y-axis is the tuning year, the color is the q2
fig, ax = plt.subplots(figsize=(30, 25))
q2_array = []
for tuning_year in tuning_years_list:
    q2_array.append(tuning_year_dict[tuning_year]["q2"])

## data clip the q2_array only keep the q2 values greater than -1
q2_array = np.clip(q2_array, -1, 1)

sns.heatmap(q2_array, cmap='PiYG', annot=True, annot_kws={"size": 15}, fmt=".2f", linewidths=0.5, ax=ax, center=0)
plt.xlabel("Predicting Year", fontsize=22)
plt.ylabel("SVM Model Using Best Hyperparameter for Predicting Year ____", fontsize=22)
plt.title("Q2 Heatmap for SVM Hyperparameter Tuning", fontsize=25)

## set the x ticks at the center of the cell
ax.set_xticks(np.arange(len(x_axis_years)) + 0.5, minor=False)
## set the y ticks at the center of the cell
ax.set_yticks(np.arange(len(tuning_years_list)) + 0.5, minor=False)

## set the x and y axis tick labels
ax.set_xticklabels(x_axis_years, minor=False, fontsize=18)
ax.set_yticklabels(tuning_years_list, minor=False, fontsize=18)

## rotate the x-axis labels
plt.xticks(rotation=45)
## compact the layout
plt.tight_layout()
plt.savefig("/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/result/plots/hyperparameter_tuning_plots/q2_tuning_year_heatmap_with_daylight.png")
plt.close()

## plot heatmap where the x-axis is the x_axis_years and the y-axis is the tuning year, the color is the rmse
fig, ax = plt.subplots(figsize=(30, 25))
rmse_array = []
for tuning_year in tuning_years_list:
    rmse_array.append(tuning_year_dict[tuning_year]["rmse"])

sns.heatmap(rmse_array, cmap='Reds', annot=True, annot_kws={"size": 15}, fmt=".2f", linewidths=0.5, ax=ax)
plt.xlabel("Predicting Year", fontsize=22)
plt.ylabel("SVM Model Using Best Hyperparameter for Predicting Year ____", fontsize=22)
plt.title("RMSE Heatmap for SVM Hyperparameter Tuning", fontsize=25)

## set the x ticks at the center of the cell
ax.set_xticks(np.arange(len(x_axis_years)) + 0.5, minor=False)
## set the y ticks at the center of the cell
ax.set_yticks(np.arange(len(tuning_years_list)) + 0.5, minor=False)

## set the x and y axis tick labels
ax.set_xticklabels(x_axis_years, minor=False, fontsize=18)
ax.set_yticklabels(tuning_years_list, minor=False, fontsize=18)

## rotate the x-axis labels
plt.xticks(rotation=45)
## compact the layout
plt.tight_layout()

plt.savefig("/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/result/plots/hyperparameter_tuning_plots/rmse_tuning_year_heatmap_with_daylight.png")
plt.close()
