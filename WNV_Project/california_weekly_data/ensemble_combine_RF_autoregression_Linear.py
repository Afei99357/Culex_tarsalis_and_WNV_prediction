import pandas as pd
# import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble, metrics


# Load the dataset into a Pandas DataFrame
df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
                 "add_0_for_no_wnv/cali_week_wnnd_multi_years_all_features_mosquitoes.csv", index_col=0)

# drop columns that are not features and drop target
df = df.drop(["State",
              # "County",
              "Year", 'Month', "County_Seat_Latitude", "County_Seat_Longitude", "FIPS",
                  # "Human_WNND_Count",
                  "Human_WNND_Rate",
                  # "Human_WNF_Count",
                  # "Total_WNV_Rate"
                  "WNV_Mos",
                  # "Population"
              ], axis=1)

# df = df.dropna()

df_county = df.pop("County")

### train and test data ####
# convert "Date" column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Split the data into training and testing sets
train_rf = df[df['Date'] < '2011-01-01']
test_rf = df[(df['Date'] >= '2011-01-01')]

# Get labels
train_rf_labels = train_rf.pop("Human_WNND_Count").values
test_rf_labels = test_rf.pop("Human_WNND_Count").values

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


########### linear model #############
## choose the columns that are not date and county
df_linear = df[['Date', 'Human_WNND_Count']]

df_linear.loc[:, 'County'] = df_county.values

# # Convert Year and Month to datetime format
# df_linear['Date'] = pd.to_datetime(df_linear['Year'].astype(str) + '-' + df_linear['Month'].astype(str), format='%Y-%m')
# df_linear = df_linear.drop(['Year', 'Month'], axis=1)

# Create a new dataframe with columns with Date and County contains the lagged values with shift 3, 2, 1 month values of Human_WNND_Count
df_lagged = pd.DataFrame()
for i in range(1, 3):
    df_lagged['t-' + str(i)] = df_linear.groupby('County')['Human_WNND_Count'].shift(i)

df_lagged['t'] = df_linear['Human_WNND_Count']
df_lagged['Date'] = df_linear['Date']
df_lagged['County'] = df_linear['County']

df_lagged = df_lagged.dropna()

# Split the data into training and testing sets
train = df_lagged[df_lagged['Date'] < '2011-01-01']
test = df_lagged[(df_lagged['Date'] >= '2011-01-01')]


train = train.drop(['Date', 'County'], axis=1)
test_date = test.pop('Date')
test_county = test.pop('County')

# Fit the auto-regressive linear model
X_train_linear = train.drop(['t'], axis=1)
y_train_linear = train['t']
X_test_linear = test.drop(['t'], axis=1)
y_test_linear = test['t']

# normalize the data
scaler = StandardScaler()
X_train_linear = scaler.fit_transform(X_train_linear)
X_test_linear = scaler.transform(X_test_linear)

# train the linear regression model
linear_regression = LinearRegression()
linear_regression.fit(X_train_linear, y_train_linear)

# Predict the values for the testing set
y_pred_linear = linear_regression.predict(X_test_linear)

########## linear model end ##########

#### combine the two models ####
# combine the two models
y_pred = y_predict_rf * 0.5 + y_pred_linear * 0.5

## create a dataframe to store the results
df_result = pd.DataFrame()
df_result['Date'] = test_date
df_result['County'] = test_county
df_result['Predicted'] = y_pred
df_result['Actual Counts'] = y_test_linear

# save the results
df_result.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
                 "auto_linear_rf_combine_model_results_count.csv")

# get the mse
mse = metrics.mean_squared_error(test_rf_labels, y_pred)

# get the r2 score
r2 = metrics.r2_score(test_rf_labels, y_pred)

# for comparison, get the mse and r2 score of a fake model that always predict the mean of the target
fake_model_mse = metrics.mean_squared_error(test_rf_labels, [y_train_linear.mean()] * len(test_rf_labels))
fake_model_r2 = metrics.r2_score(test_rf_labels, [y_train_linear.mean()] * len(test_rf_labels))

# # print the mse and r2 score
# check the standard deviation of the target
print("The standard deviation of the target is: ", df["Human_WNND_Count"].std())
print("The mean squared error of ensemble Model verse Fake Model: {:.03}, vs {:.03}".format(mse, fake_model_mse))
print("The r2 score of ensemble Model verse Fake Model: {:.03}, vs {:.03}".format(r2, fake_model_r2))
