from sklearn.svm import SVR
from sklearn import metrics
import pandas as pd
import shap
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, tpe, hp


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

## scale the data
scaler = StandardScaler()
train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
test = pd.DataFrame(scaler.transform(test), columns=test.columns)

# Get the column names
train_column_names = train.columns
test_column_names = test.columns

## read the best hyperparameters from the file
best_hyperparameters = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/"
                                   "CA_13_county_dataset/result/plots/SVM/hyperparameter_tuning_plots/"
                                   "hyperparameter_tuning_svm_impute_0_with_daylight_best_hyperparameters.csv",
                                      index_col=False, header=0)

## create a dataframe to store the tuning year, q2 and RMSE and hyperparameters
tuning_year_q2_rmse = pd.DataFrame(columns=["tuning_year", "q2", "RMSE", "C", "epsilon", "gamma", "kernel"])

## create a list to store the tuning year, q2 and RMSE
tuning_year_q2_rmse_list = []

for index, row in best_hyperparameters.iterrows():
    hyperparameter_tuning_year = row['tuning_year']
    C = float(row['C'])
    epsilon = float(row['epsilon'])
    gamma = row['gamma']
    kernel = row['kernel']

    ## with daylight using 2018
    model = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel=kernel)

    model.fit(train, train_labels)

    # Predict the test data
    predictions = model.predict(test)

    # Calculate the Q^2 and RMSE
    q2 = metrics.r2_score(test_labels, predictions)

    # Calculate the RMSE
    rmse = np.sqrt(metrics.mean_squared_error(test_labels, predictions))

    ## append the tuning year, q2 and RMSE to the list
    tuning_year_q2_rmse_list.append([hyperparameter_tuning_year, q2, rmse, C, epsilon, gamma, kernel])

    ## print the Q^2 and RMSE with tuning year
    print(f"tuning year: {hyperparameter_tuning_year}, q2: {q2}, RMSE: {rmse}, ")

## append the tuning year, q2 and RMSE to the dataframe
tuning_year_q2_rmse = pd.DataFrame(tuning_year_q2_rmse_list, columns=["tuning_year", "q2", "RMSE", "C", "epsilon", "gamma", "kernel"])

## save the tuning year, q2 and RMSE to a csv file
tuning_year_q2_rmse.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/result/plots/SVM/hyperparameter_tuning_plots/"
                           "hyperparameter_tuning_svm_impute_0_q2_rmse.csv", index=False)

#
# def f(X):
#     return model.predict(X)

# ## shap values
# explainer = shap.Explainer(f, test)(test)
#
# ## plot global shap values
# ## plot global bar plot where the global importance of each feature is taken to be the mean absolute value for that feature over all the given samples.
# plt.figure(figsize=(30, 10))
#
# shap.plots.bar(explainer, show=False, max_display=18)
# plt.tight_layout()
# plt.savefig(f"/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/result/plots/train_before_2019_01_01_predict_after_2019_01_01/using_2018_model_best_hypoerparameter/shap_plots/svm_global_shap_plot_predict_19_to_23_with_daylight.png")
# plt.close()
# #######
#
# ### plot local shap values, individual shap values
# ## for each sample in the x_test, get the year, month and county information for output file
# for i in range(len(test)):
#     plt.figure(figsize=(60, 20))
#
#     ## adding padding to the plot
#     plt.subplots_adjust(left=0.4, right=0.6, top=0.9, bottom=0.1)
#
#     shap.plots.bar(explainer[i], show=False, max_display=17)
#     month = test_month_list[i]
#     year = test_year_list[i]
#     county = test_FIPS_list[i]
#     plt.tight_layout()
#     plt.savefig(f"/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/result/plots/train_before_2019_01_01_predict_after_2019_01_01/using_2018_model_best_hypoerparameter/shap_plots/individual_sample_shap_predict_19_to_23_withdaylight_hours/svm_local_shap_plot_{year}_{month}_{county}.png")
#     plt.close()


