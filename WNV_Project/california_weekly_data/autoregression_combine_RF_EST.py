import pandas as pd
# import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble, metrics

from sklearn.ensemble import HistGradientBoostingRegressor


# Load the dataset into a Pandas DataFrame
df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
                 "add_0_for_no_wnv/cali_week_wnnd_multi_years_all_features_mosquitoes.csv", index_col=0)

# drop columns that are not features and drop target
df = df.drop(["State",
              # "County",
              "Year", 'Month', "County_Seat_Latitude", "County_Seat_Longitude", "FIPS",
              "Human_WNND_Count",
              # "WNV_Mos",
              # "Human_WNND_Rate",
              # "Population"
              ], axis=1)

df = df.dropna()

df_county = df.pop("County")

### train and test data ####
# convert "Date" column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Split the data into training and testing sets
train_rf = df[df['Date'] < '2012-01-01']
test_rf = df[(df['Date'] >= '2012-01-01') & (df['Date'] < '2013-01-01')]

# Get labels
train_rf_labels = train_rf.pop("Human_WNND_Rate").values
test_rf_labels = test_rf.pop("Human_WNND_Rate").values

train_rf.pop("Date")
test_rf.pop("Date")

# get the column names
train_column_names = train_rf.columns
test_column_names = test_rf.columns

########### Random Forest Classifier #############
rf = ensemble.RandomForestRegressor(n_estimators=4, max_depth=3, max_features=0.7, n_jobs=-1)
rf.fit(train_rf, train_rf_labels)
y_predict_rf = rf.predict(test_rf)
########## random forest model end #############


########### Second RF model #############
## choose the columns that are not date and county
df_linear = df[['Date', 'Human_WNND_Rate', 'WNV_Mos']]

df_linear.loc[:, 'County'] = df_county.values

# # Convert Year and Month to datetime format
# df_linear['Date'] = pd.to_datetime(df_linear['Year'].astype(str) + '-' + df_linear['Month'].astype(str), format='%Y-%m')
# df_linear = df_linear.drop(['Year', 'Month'], axis=1)

# Create a new dataframe with columns with Date and County contains the lagged values with shift 3, 2, 1 month values of Human_WNND_Count
df_lagged = pd.DataFrame()
for i in range(1, 3):
    df_lagged['WNND-' + str(i)] = df_linear.groupby('County')['Human_WNND_Rate'].shift(i)
    df_lagged['Mos-' + str(i)] = df_linear.groupby('County')['WNV_Mos'].shift(i)

df_lagged['WNND_Current'] = df_linear['Human_WNND_Rate']
df_lagged['Mos_Current'] = df_linear['WNV_Mos']
df_lagged['Date'] = df_linear['Date']
df_lagged['County'] = df_linear['County']

# df_lagged = df_lagged.dropna()

# Split the data into training and testing sets
train = df_lagged[df_lagged['Date'] < '2012-01-01']
test = df_lagged[(df_lagged['Date'] >= '2012-01-01') & (df_lagged['Date'] < '2013-01-01')]


train = train.drop(['Date', 'County'], axis=1)
test_date = test.pop('Date')
test_county = test.pop('County')

# Fit the auto-regressive linear model
X_train_est = train.drop(['WNND_Current'], axis=1)
y_train_est = train['WNND_Current']
X_test_est = test.drop(['WNND_Current'], axis=1)
y_test_est = test['WNND_Current']

########### Random Forest Classifier #############
est = ensemble.HistGradientBoostingRegressor(max_iter=1000, max_depth=3, max_leaf_nodes=10, learning_rate=0.1)
est.fit(X_train_est, y_train_est)
########## random forest model end #############

# Predict the values for the testing set
y_predict_est = est.predict(X_test_est)
########## linear model end ##########

#### combine the two models ####
# combine the two models
y_pred = y_predict_rf * 0.5 + y_predict_est * 0.5

## create a dataframe to store the results
df_result = pd.DataFrame()
df_result['Date'] = test_date
df_result['County'] = test_county
df_result['Predicted'] = y_pred
df_result['Actual Counts'] = y_test_est

# save the results
df_result.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
                 "auto_linear_est_combine_model_results.csv")

# get the mse
mse = metrics.mean_squared_error(test_rf_labels, y_pred)

# get the r2 score
r2 = metrics.r2_score(test_rf_labels, y_pred)

# for comparison, get the mse and r2 score of a fake model that always predict the mean of the target
fake_model_mse = metrics.mean_squared_error(test_rf_labels, [y_train_est.mean()] * len(test_rf_labels))
fake_model_r2 = metrics.r2_score(test_rf_labels, [y_train_est.mean()] * len(test_rf_labels))

# # print the mse and r2 score
# check the standard deviation of the target
print("The standard deviation of the target is: ", df["Human_WNND_Rate"].std())
print("The mean squared error of Linear Regression Model verse Fake Model: {:.03}, vs {:.03}".format(mse, fake_model_mse))
print("The r2 score of Linear Regression Model verse Fake Model: {:.03}, vs {:.03}".format(r2, fake_model_r2))
