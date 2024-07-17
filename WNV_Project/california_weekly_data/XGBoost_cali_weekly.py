import xgboost as xgb
import pandas as pd
from sklearn import metrics
from sklearn.metrics import mean_squared_error

# Load data into pandas DataFrame
# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
                 "add_0_for_no_wnv/cali_week_wnnd_multi_years_all_features_mosquitoes.csv", index_col=0)
# drop columns that are not features and drop target
data = data.drop(["State",
              "County",
              "Year", 'Month', "County_Seat_Latitude", "County_Seat_Longitude", "FIPS",
                  # "Human_WNND_Count",
                  "Human_WNND_Rate",
                  "WNV_Mos_Count",
                  # "Population"
              ], axis=1)
# drop row where "WNV_Mos" columns value is NaN
data = data.dropna(subset=["Human_WNND_Count"])

# convert "Date" column to datetime
data["Date"] = pd.to_datetime(data["Date"])

## get the "Date" before 2012-05-01 as train data
# train = data[(data["Date"] > "2010-01-01") & (data["Date"] < "2012-01-01")]
# test = data[(data["Date"] >= "2012-01-01") & (data["Date"] < "2012-12-31")]

### 111 ####
### train and test data ####
X_train = data[(data["Date"] < "2010-01-01")]
X_test = data[(data["Date"] >= "2010-01-01")]

# Get labels
y_train = X_train.pop("Human_WNND_Count").values
y_test = X_test.pop("Human_WNND_Count").values

X_train.pop("Date")
X_test.pop("Date")

# get the column names
train_column_names = X_train.columns
test_column_names = X_test.columns

# Define XGBoost model
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.5,
    colsample_bytree=0.5
)

# Train XGBoost model
model.fit(X_train, y_train)

# Generate predictions on testing data
y_pred = model.predict(X_test)

# Evaluate performance using mean squared error
# ## get the mse, r2 score
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

# for comparison, get the mse and r2 score of a fake model that always predict the mean of the target
fake_model_mse = metrics.mean_squared_error(y_test, [y_train.mean()] * len(y_test))
fake_model_r2 = metrics.r2_score(y_test, [y_train.mean()] * len(y_test))

# #print the mse and r2 score
# check the standard deviation of the target
print("The standard deviation of the target is: ", data["Human_WNND_Count"].std())
print("The mean squared error of Histogram-based Gradient Boosting Regression Tree Model verse Fake Model: {:.03}, vs {:.03}".format(mse, fake_model_mse))
print("The r2 score of Histogram-based Gradient Boosting Regression Tree Model verse Fake Model: {:.03}, vs {:.03}".format(r2, fake_model_r2))
