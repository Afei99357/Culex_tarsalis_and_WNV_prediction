from sklearn.svm import SVR
from sklearn import metrics
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
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
train = data[data['Year'] < 2019].copy()
# test = data[(data['Year'] >= 2019)].copy()

## get test data only for year in 2019 and 2023
test = data[data['Year'].isin([2019])]

## bootstraping training data for 1000 iterations
n_iterations = 1000
n_size = int(len(train))
train_samples = []

for i in range(n_iterations):
    sample = train.sample(n=n_size, replace=True)
    train_samples.append(sample)

## boostraping to train the svm model and predict the testing data, get the q2 and mse
q2_list = []
rmse_list = []

for i in range(n_iterations):
    print(f"iteration: {i}")
    x_train = train_samples[i]
    x_test = test.copy()

    # Get labels
    y_train = x_train.pop("Human_Disease_Count").values
    y_test = x_test.pop("Human_Disease_Count").values

    ## Remove unnecessary columns
    x_train.drop(["Month", "FIPS", "Year"], axis=1, inplace=True)
    x_test.drop(["Month", "FIPS", "Year"], axis=1, inplace=True)

    # Scale the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # # Train the model with svm using 2018 tuned hyperparameters
    # model = SVR(C=8.967266674728009, epsilon=0.10424919467608322, gamma='auto', kernel='rbf')

    # Train the model with svm using 2009 tuned hyperparameters
    model = SVR(C=0.660013053582507, epsilon=0.188805559508538, gamma='auto', kernel='poly')

    model.fit(x_train, y_train)

    # Predict the test data
    predictions = model.predict(x_test)

    # Calculate the q2 and rmse
    q2 = metrics.r2_score(y_test, predictions)
    mse = metrics.mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    q2_list.append(q2)
    rmse_list.append(rmse)

## calculate confidence interval for q2 and rmse
q2_mean = round(np.mean(q2_list), 2)
rmse_mean = round(np.mean(rmse_list), 2)

## define the confidence interval
def confidence_interval(data, alpha=0.05):
    lower_bound = np.percentile(data, 100 * alpha / 2, axis=0)
    upper_bound = np.percentile(data, 100 * (1 - alpha / 2), axis=0)
    return round(lower_bound, 2), round(upper_bound, 2)

q2_lower, q2_upper = confidence_interval(q2_list)

mse_lower, mse_upper = confidence_interval(rmse_list)

## plot the q2 and mse separately with confidence interval
plt.figure(figsize=(10, 5))
plt.hist(q2_list, bins=30, color='blue', alpha=0.5)
plt.axvline(q2_mean, color='red', linestyle='dashed', linewidth=2, label=f'mean Q2: {q2_mean}')
plt.axvline(q2_lower, color='purple', linestyle='dashed', linewidth=2, label=f'95% confidence interval lower bound: {q2_lower}')
plt.axvline(q2_upper, color='black', linestyle='dashed', linewidth=2, label=f'95% confidence interval upper bound: {q2_upper}')

## add legend
plt.legend(loc='upper left')

plt.title("Q2 distribution")
plt.savefig("/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/result/plots/train_before_2019_01_01_predict_after_2019_01_01/using_2009_model_best_hyperparameter/bootstrapping_svm_q2_distribution_remove_20_21_22_23.png")
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(rmse_list, bins=30, color='blue', alpha=0.5)
plt.axvline(np.mean(rmse_list), color='red', linestyle='dashed', linewidth=2, label=f'mean RMSE: {rmse_mean}')
plt.axvline(mse_lower, color='purple', linestyle='dashed', linewidth=2, label=f'95% confidence interval lower bound: {mse_lower}')
plt.axvline(mse_upper, color='black', linestyle='dashed', linewidth=2, label=f'95% confidence interval upper bound: {mse_upper}')

## add legend
plt.legend(loc='upper right')

plt.title("RMSE distribution")
plt.savefig("/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/result/plots/train_before_2019_01_01_predict_after_2019_01_01/using_2009_model_best_hyperparameter/bootstrapping_svm_rmse_distribution_remove_20_21_22_23.png")
plt.show()


