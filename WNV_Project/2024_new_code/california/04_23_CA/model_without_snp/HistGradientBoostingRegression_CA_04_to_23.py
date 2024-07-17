from sklearn import ensemble, metrics
import pandas as pd
import numpy as np



# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/CA_human_data_2004_to_2023_final_no_impute_0.csv",
                   index_col=False,
                   header=0)

## get the Date column to a new dataframe
date = data.pop("Date")

# drop columns that are not features and drop target
data = data.drop([
    "Year",
    # 'Month',
    "County",
    # 'FIPS',
    "Latitude",
    "Longitude",
    "Total_Bird_WNV_Count",
    "Mos_WNV_Count",
    "Horse_WNV_Count",
], axis=1)

## drop the columns if all the values in the columns are the same or all nan
data = data.dropna(axis=1, how='all')

## reindex the data
data = data.reset_index(drop=True)

## print 0 variance columns
print(data.columns[data.var() == 0])

## check if any columns has zero variance and drop the columns
data = data.loc[:, data.var() != 0]

## add the Date column back to the data
data["Date"] = date

# convert "Date" column to datetime
data["Date"] = pd.to_datetime(data["Date"])

### train and test data ####
train = data[data['Date'] < '2020-01-01']
## get the test data in 2020
test = data[data['Date'] >= '2020-01-01']

## impute the nan values with the mean of the column
# ## adding an empty column average_cases column to both train and test data
train["average_human_case_monthly"] = ""

# ## find unique years in train data
years = train["Date"].dt.year.unique()

## calculate the average human cases monthly for each FIPS and month using train
train["average_human_case_monthly"] = train.groupby(["FIPS", "Month"])["Human_Disease_Count"].transform("sum") / len(years)

## use the train dataset to impute 0
train.loc[train["Human_Disease_Count"].isna(), "Human_Disease_Count"] = train["average_human_case_monthly"]

## drop rows if has nan values for both train and test data
train = train.dropna().reset_index(drop=True)
test = test.dropna().reset_index(drop=True)

# Get labels
train_labels = train.pop("Human_Disease_Count").values
test_labels = test.pop("Human_Disease_Count").values

## remove unnecessary columns
train.drop(["Month", "FIPS", "Date", "average_human_case_monthly"], axis=1, inplace=True)
test.drop(["Month", "FIPS", "Date"], axis=1, inplace=True)

# get the column names
train_column_names = train.columns
test_column_names = test.columns

######################### HistGradientBoostingRegressor ######################################
######################### RF ######################################
## HGBR
hgbr = ensemble.HistGradientBoostingRegressor(max_depth=20)

hgbr.fit(train, train_labels)

y_predict = hgbr.predict(test)

# ## if y_predict is negative, set it to 0
# y_predict = np.where(y_predict < 0, 0, y_predict)

# store test labels and predicted labels in a dataframe
df = pd.DataFrame({"test_labels": test_labels, "y_predict": y_predict})
df.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/result/Histogram-Based Gradient Boosting Ensembles/EST_cali_03_to_24_result.csv")

# ## get the mse, r2 score
mse = metrics.mean_squared_error(test_labels, y_predict)
r2 = metrics.r2_score(test_labels, y_predict)
# for comparison, get the mse and r2 score of a fake model that always predict the mean of the target
fake_model_mse = metrics.mean_squared_error(test_labels, [train_labels.mean()] * len(test_labels))
fake_model_r2 = metrics.r2_score(test_labels, [train_labels.mean()] * len(test_labels))

# #print the mse and r2 score
# check the standard deviation of the target
print("The standard deviation of the target is: ", data["Human_Disease_Count"].std())
print("The mean squared error of Histogram-based Gradient Boosting Regression Tree Model verse Fake Model: {:.03}, vs {:.03}".format(mse, fake_model_mse))
print("The r2 score of Histogram-based Gradient Boosting Regression Tree Model verse Fake Model: {:.03}, vs {:.03}".format(r2, fake_model_r2))
