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

tuning_year = 2006
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

        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100)

        # Map the best indices back to the corresponding string values
        best['kernel'] = kernel_map[int(best['kernel'])]
        best['gamma'] = gamma_map[int(best['gamma'])]

        ## update the best hyperparameters
        best_hyperparameters.append(best)

print("best_hyperparameters: ", best_hyperparameters)
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

        ## scale the data
        scaler = StandardScaler()
        train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
        test = pd.DataFrame(scaler.transform(test), columns=test.columns)

        # Get the column names
        train_column_names = train.columns
        test_column_names = test.columns

        # SVM
        svm = SVR(**best_hyperparameters[0])
        # svm = SVR(C=9.98593930616599, epsilon=0.10904839812283, gamma='auto', kernel='rbf')
        svm.fit(train, train_labels)

        y_predict = svm.predict(test)

        ## get the mean of the predicted values
        y_predict_mean_list.append(np.mean(y_predict))

        ## get the mean of the test labels
        test_labels_mean_list.append(np.mean(test_labels))
        #
        # def f(X):
        #     return svm.predict(X)
        #
        # # x_shap = shap.sample(test, 150)
        #
        # ## shap values
        # explainer = shap.Explainer(f, test)(test)
        #
        # # plt.close()
        # ## plot global bar plot where the global importance of each feature is taken to be the mean absolute value for that feature over all the given samples.
        # plt.figure(figsize=(30, 10))
        #
        # shap.plots.bar(explainer, show=False, max_display=17)
        # plt.tight_layout()
        # plt.savefig(f"/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/result/plots/shap_summary_accumulated_svm/svm_global_shap_plot_{year}.png")


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

## plot the mean of the predicted values and the mean of the test labels together
plt.figure(figsize=(10, 5))
unique_years = [i for i in years if i >= 2005]
plt.plot(unique_years, y_predict_mean_list, label="mean of the predicted values")
plt.plot(unique_years, test_labels_mean_list, label="mean of the test labels")
plt.xlabel("Year")
plt.ylabel("Mean")

plt.legend()

plt.title("Mean of the predicted values and the mean of the test labels")
plt.tight_layout()

plt.show()

print("mse_list: ", mse_list)
print("r2_list: ", r2_list)