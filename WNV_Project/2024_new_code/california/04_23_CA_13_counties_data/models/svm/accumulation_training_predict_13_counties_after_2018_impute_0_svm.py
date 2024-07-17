from sklearn.svm import SVR
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


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

## get the training and testing data
x_train = data[data['Year'] < 2019].copy()

## get test data only for year in 2019 and 2023
x_test = data[data['Year'].isin([2019, 2020, 2021, 2022, 2023])]

# Get labels
y_train = x_train.pop("Human_Disease_Count").values
y_test = x_test.pop("Human_Disease_Count").values

## Remove unnecessary columns
x_train.drop(["Month", "FIPS", "Year"], axis=1, inplace=True)
# x_test.drop(["Month", "Year"], axis=1, inplace=True)
test_FIPS_list = x_test.pop("FIPS").values
test_month_list = x_test.pop("Month").values
test_year_list = x_test.pop("Year").values

## get the column names
column_names = x_train.columns

# Scale the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train the model with svm using tuned hyperparameters
model = SVR(C=8.967266674728009, epsilon=0.10424919467608322, gamma='auto', kernel='rbf')

## adding the column names back to the data
x_train = pd.DataFrame(x_train, columns=column_names)
x_test = pd.DataFrame(x_test, columns=column_names)

model.fit(x_train, y_train)

# Predict the test data
predictions = model.predict(x_test)

## caluclate the q2 and mse
q2 = model.score(x_test, y_test)
mse = ((predictions - y_test) ** 2).mean()
rmse = mse ** 0.5

## print the q2 and mse
print(f"q2: {q2}")
print(f"RMSE: {rmse}")

## output the predictions
output = pd.DataFrame({ "FIPS": test_FIPS_list, "Month": test_month_list, "Year": test_year_list, "Human_Disease_Count": predictions, "True_Human_Disease_Count": y_test})

## save the output to a csv file
output.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/result/svm_predictions_13_counties_CA_19_to_23.csv", index=False)
#
# def f(X):
#     return model.predict(X)
#
# ## shap values
# explainer = shap.Explainer(f, x_test)(x_test)
#
# ## plot global shap values
# ## plot global bar plot where the global importance of each feature is taken to be the mean absolute value for that feature over all the given samples.
# plt.figure(figsize=(30, 10))
#
# shap.plots.bar(explainer, show=False, max_display=17)
# plt.tight_layout()
# plt.savefig(f"/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/result/plots/train_before_2019_01_01_predict_after_2019_01_01/shap_plots/svm_global_shap_plot_predict_19_to_23.png")
# plt.close()
# #######
#
# ### plot local shap values, individual shap values
# ## for each sample in the x_test, get the year, month and county information for output file
# for i in range(len(x_test)):
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
#     plt.savefig(f"/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/result/plots/train_before_2019_01_01_predict_after_2019_01_01/shap_plots/individual_sample_shap_predict_19_to_23/svm_local_shap_plot_{year}_{month}_{county}.png")
#     plt.close()
