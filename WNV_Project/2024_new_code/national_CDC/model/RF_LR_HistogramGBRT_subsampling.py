from sklearn import ensemble, metrics
import pandas as pd
from sklearn.utils import resample

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/"
                   "human/cdc_human_1999_to_2023/WNV_human_and_non_human_yearly_climate_demographic_bird.csv", index_col=0)

## remove any space and comma in Population column
data["Population"] = data["Population"].str.replace(",", "").str.strip()

## convert Population column to numeric
data["Population"] = pd.to_numeric(data["Population"], errors='coerce')

# select the columns after column Date as predictors
date_index = data.columns.get_loc("Date")

# Select columns after the "Date" column as predictors
data_pred = data.iloc[:, date_index+1:]

# get Year column from data and add it to data_pred
data_pred['Year'] = data['Year']

## add target column
data_pred["Neuroinvasive_disease_cases"] = data["Neuroinvasive_disease_cases"]

# drop nan values
data_pred = data_pred.dropna()

## reset index
data_pred = data_pred.reset_index(drop=True)


## define subsampling function
def balance_classes(data, target_column, seed=123):
    # Separate majority and minority classes
    data_majority = data[data[target_column] == 0]
    data_minority = data[data[target_column] > 0]

    # Downsample majority class or Upsample minority class
    if len(data_majority) > len(data_minority):
        data_majority_downsampled = resample(data_majority,
                                             replace=False,  # sample without replacement
                                             n_samples=len(data_minority),  # to match minority class
                                             random_state=seed)  # reproducible results
        # Combine minority class with downsampled majority class
        data_balanced = pd.concat([data_majority_downsampled, data_minority])
    else:
        data_minority_upsampled = resample(data_minority,
                                           replace=True,  # sample with replacement
                                           n_samples=len(data_majority),  # to match majority class
                                           random_state=seed)  # reproducible results
        # Combine majority class with upsampled minority class
        data_balanced = pd.concat([data_minority_upsampled, data_majority])

    return data_balanced

## balance the classes
data_pred = balance_classes(data_pred, "Neuroinvasive_disease_cases")

### train and test data ####
train = data_pred[(data_pred["Year"] < 2018)]
test = data_pred[(data_pred["Year"] >= 2018)]

# Get labels
train_labels = train.pop("Neuroinvasive_disease_cases").values
test_labels = test.pop("Neuroinvasive_disease_cases").values

## remove time column
train.pop("Year")
test.pop("Year")

# get the column names
train_column_names = train.columns
test_column_names = test.columns

## RUN THE MODEL 100 times and record the mse and r2 score
rf_mse_list = []
rf_r2_list = []
est_mse_list = []
est_r2_list = []
for i in range(100):
    ######################### RF ######################################
    ## Random Forest Classifier
    rf = ensemble.RandomForestRegressor(n_estimators=4, max_depth=3, max_features=0.7, n_jobs=-1)

    rf.fit(train, train_labels)

    y_predict = rf.predict(test)

    ## HistGradientBoostingRegressor
    est = ensemble.HistGradientBoostingRegressor(max_iter=1000, max_depth=2, max_leaf_nodes=5, learning_rate=0.1)

    est.fit(train, train_labels)

    y_predict_est = est.predict(test)

    # store test labels and predicted labels in a dataframe
    df = pd.DataFrame({"test_labels": test_labels, "y_predict": y_predict})
    df.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/result/RF_HistogramGBRT_yearly_subsampling/human_yearly_cdc_rf_subsampling.csv")

    df_est = pd.DataFrame({"test_labels": test_labels, "y_predict": y_predict_est})
    df_est.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/result/RF_HistogramGBRT_yearly_subsampling/human_yearly_cdc_est_subsampling.csv")

    # ## get the mse, r2 score
    mse = metrics.mean_squared_error(test_labels, y_predict)
    r2 = metrics.r2_score(test_labels, y_predict)

    mse_est = metrics.mean_squared_error(test_labels, y_predict_est)
    r2_est = metrics.r2_score(test_labels, y_predict_est)

    rf_mse_list.append(mse)
    rf_r2_list.append(r2)
    est_mse_list.append(mse_est)
    est_r2_list.append(r2_est)


## print average mse and r2 score
print("average mse: ", sum(rf_mse_list)/len(rf_mse_list))
print("average r2: ", sum(rf_r2_list)/len(rf_r2_list))

print("HistogramGBRT average mse: ", sum(est_mse_list)/len(est_mse_list))
print("HistogramGBRT average r2: ", sum(est_r2_list)/len(est_r2_list))

### run linear regression model 100 times and record the mse and r2 score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

## run the linear regression model and record the mse and r2 score
# train the linear regression model

# normalize the data
scaler = StandardScaler()
lr_train = scaler.fit_transform(train)
lr_test = scaler.transform(test)

# train the linear regression model
linear_regression = LinearRegression()
linear_regression.fit(lr_train, train_labels)

linear_regression = LinearRegression()
linear_regression.fit(lr_train, train_labels)

y_predict = linear_regression.predict(lr_test)

# calculate the mse and r2 score
lr_mse = metrics.mean_squared_error(test_labels, y_predict)

lr_r2 = metrics.r2_score(test_labels, y_predict)

## print the mse and r2 score
print("lr_mse: ", lr_mse)
print("lr_r2: ", lr_r2)

### plot boxplot of mse and r2 score
import matplotlib.pyplot as plt

plt.boxplot([rf_mse_list, rf_r2_list, [lr_mse], [lr_r2], est_mse_list, est_r2_list], labels=["rf_mse", "rf_r2", "lr_mse", "lr_r2", "est_mse", "est_r2"])

plt.show()


