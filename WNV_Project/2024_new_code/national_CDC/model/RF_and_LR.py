from sklearn import ensemble, metrics
import pandas as pd

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
for i in range(100):
    ######################### RF ######################################
    ## Random Forest Classifier
    rf = ensemble.RandomForestRegressor(n_estimators=4, max_depth=3, max_features=0.7, n_jobs=-1)

    rf.fit(train, train_labels)

    y_predict = rf.predict(test)

    # store test labels and predicted labels in a dataframe
    df = pd.DataFrame({"test_labels": test_labels, "y_predict": y_predict})
    df.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/result/RF/human_yearly_cdc_rf.csv")

    # ## get the mse, r2 score
    mse = metrics.mean_squared_error(test_labels, y_predict)
    r2 = metrics.r2_score(test_labels, y_predict)

    rf_mse_list.append(mse)
    rf_r2_list.append(r2)

## print average mse and r2 score
print("average mse: ", sum(rf_mse_list)/len(rf_mse_list))
print("average r2: ", sum(rf_r2_list)/len(rf_r2_list))

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

y_predict = linear_regression.predict(lr_test)

# calculate the mse and r2 score
lr_mse = metrics.mean_squared_error(test_labels, y_predict)

lr_r2 = metrics.r2_score(test_labels, y_predict)

## print the mse and r2 score
print("lr_mse: ", lr_mse)
print("lr_r2: ", lr_r2)

### plot boxplot of mse and r2 score
import matplotlib.pyplot as plt

plt.boxplot([rf_mse_list, rf_r2_list, [lr_mse], [lr_r2]], labels=["rf_mse", "rf_r2", "lr_mse", "lr_r2"])



