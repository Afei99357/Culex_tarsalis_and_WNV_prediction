import pandas as pd
# import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble, metrics


# Load the dataset into a Pandas DataFrame
df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/data/cali_week_impute_monthly_mean_value_with_allele_frequency_RDA_after_drop_high_cor_and_low_target_cor.csv",
                   index_col=False,
                   header=0)

# drop columns that are not features and drop target
df = df.drop([
    # "County",
    "Year",
    'Month',
    # 'FIPS',
    'State',
    "Latitude",
    "Longitude",
    # 'Population',
    'Land_Area_2010',
    "Poverty_Estimate_All_Ages",
    # "American Goldfinch",
    # "House Finch",
    # "House Sparrow",
    # 'American Crow',
    # 'American Robin',
    # "Human_WNND_Rate",
    # "WNV_Mos_Count",
    # "WNV_Corvid_Count",
    # "WNV_NonCorvid_Count",
    # "Total_Bird_WNV_Count",
    "incident_rate_WNV"
], axis=1)

df_fips = df.pop("FIPS")

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

df_linear.loc[:, 'FIPS'] = df_fips.values

# Create a new dataframe with columns with Date and County contains the lagged values with shift 3, 2, 1 month values of Human_WNND_Count
df_lagged = pd.DataFrame()
for i in range(1, 3):
    df_lagged['t-' + str(i)] = df_linear.groupby('FIPS')['Human_WNND_Count'].shift(i)

df_lagged['t'] = df_linear['Human_WNND_Count']
df_lagged['Date'] = df_linear['Date']
df_lagged['FIPS'] = df_linear['FIPS']

df_lagged = df_lagged.dropna()

# Split the data into training and testing sets
train = df_lagged[df_lagged['Date'] < '2011-01-01']
test = df_lagged[(df_lagged['Date'] >= '2011-01-01')]

train = train.drop(['Date', 'FIPS'], axis=1)
test_date = test.pop('Date')
test_fips = test.pop('FIPS')

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
df_result['FIPS'] = test_fips
df_result['Predicted'] = y_pred
df_result['Actual Counts'] = y_test_linear

# save the results
df_result.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/autoregression_combine_RF_and_Linear/"
                 "auto_linear_rf_combine_model_results_count_allel_frequency_all_gene_after_drop_high_correlated.csv")

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
