from sklearn import ensemble, metrics
import pandas as pd

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/data/cali_week_impute_monthly_mean_value_with_allele_frequency_RDA_after_drop_high_cor_and_low_target_cor.csv",
                   index_col=False,
                   header=0)

# drop columns that are not features and drop target
data = data.drop([
    # "County",
    "Year",
    'Month',
    'FIPS',
    'State',
    "Latitude",
    "Longitude",
    # 'Population',
    'Land_Area_2010',
    "Poverty_Estimate_All_Ages",
    "tig00000199_2572223.T",
    "tig00000379_62942.C",
    "tig00003801_23907.G"
], axis=1)

data = data.dropna()

# convert "Date" column to datetime
data["Date"] = pd.to_datetime(data["Date"])

### 111 ####
### train and test data ####
train = data[data['Date'] < '2011-01-01']
test = data[(data['Date'] >= '2011-01-01')]

# Get labels
train_labels = train.pop("Human_WNND_Count").values
test_labels = test.pop("Human_WNND_Count").values

train.pop("Date")
test.pop("Date")

# get the column names
train_column_names = train.columns
test_column_names = test.columns


## run the rf model 100 times and record the mse and r2 score
mse_list = []
r2_list = []
for i in range(100):
    ######################### RF ######################################
    ## Random Forest Classifier
    rf = ensemble.RandomForestRegressor(n_estimators=4, max_depth=3, max_features=0.7, n_jobs=-1)

    rf.fit(train, train_labels)

    y_predict = rf.predict(test)

    # ## get the mse, r2 score
    mse = metrics.mean_squared_error(test_labels, y_predict)
    r2 = metrics.r2_score(test_labels, y_predict)
    mse_list.append(mse)
    r2_list.append(r2)

## plot boxplot of mse and r2 score
import matplotlib.pyplot as plt

plt.boxplot([mse_list, r2_list], labels=["mse", "r2"])
plt.show()

#
# # store test labels and predicted labels in a dataframe
# df = pd.DataFrame({"test_labels": test_labels, "y_predict": y_predict})
# df.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/RF/RF_cali_wnnd_without_snp.csv")
#
# # ## get the mse, r2 score
# mse = metrics.mean_squared_error(test_labels, y_predict)
# r2 = metrics.r2_score(test_labels, y_predict)
# # for comparison, get the mse and r2 score of a fake model that always predict the mean of the target
# fake_model_mse = metrics.mean_squared_error(test_labels, [train_labels.mean()] * len(test_labels))
# fake_model_r2 = metrics.r2_score(test_labels, [train_labels.mean()] * len(test_labels))
#
# # #print the mse and r2 score
# # check the standard deviation of the target
# print("The standard deviation of the target is: ", data["Human_WNND_Count"].std())
# print("Without SNPs, The mean squared error of Random Forest Model verse Fake Model: {:.03}, vs {:.03}".format(mse, fake_model_mse))
# print("Without SNPs, The r2 score of Random Forest Model verse Fake Model: {:.03}, vs {:.03}".format(r2, fake_model_r2))
