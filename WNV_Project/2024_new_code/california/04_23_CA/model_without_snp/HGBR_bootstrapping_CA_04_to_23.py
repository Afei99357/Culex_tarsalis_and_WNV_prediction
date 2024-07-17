from sklearn import ensemble, metrics
from sklearn.utils import resample
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/CA_human_data_2004_to_2023_final.csv",
                   index_col=False,
                   header=0)

## get the Date column to a new dataframe
date = data.pop("Date")

# drop columns that are not features and drop target
data = data.drop([
    "Year",
    'Month',
    "County",
    'FIPS',
    "Latitude",
    "Longitude",
    "Total_Bird_WNV_Count",
    "Mos_WNV_Count",
    "Horse_WNV_Count",
    "average_human_case_monthly",
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

## drop rows if has nan values
data = data.dropna().reset_index(drop=True)

### train and test data ####
train = data[(data['Date'] > '2018-01-01') & (data['Date'] < '2023-01-01')]
## get the test data in 2023
test = data[(data['Date'] >= '2023-01-01')]
test_labels = test.pop("Human_Disease_Count").values

# remove the Date column
train.pop("Date")
test.pop("Date")

## Bootstrapping
n_bootstraps = 1000

## create a list to store the q2 and mse
q2 = []
mse = []


for _ in range(n_bootstraps):
    print(_)
    bootstrap_data = resample(train, n_samples=len(train), replace=True)

    ## reset index
    bootstrap_data = bootstrap_data.reset_index(drop=True)

    ## get the labels
    train_labels = bootstrap_data.pop("Human_Disease_Count").values

    ## HistGradientBoostingRegressor
    model = ensemble.HistGradientBoostingRegressor()

    model.fit(bootstrap_data, train_labels)

    ## predict the test data
    y_predict = model.predict(test)

    q2.append(metrics.r2_score(test_labels, y_predict))
    print(metrics.r2_score(test_labels, y_predict))
    mse.append(metrics.mean_squared_error(test_labels, y_predict))

## get the 95% confident interval
q2_lower = np.percentile(q2, 2.5)
q2_upper = np.percentile(q2, 97.5)

## plot the q2 in histogram and add the 95% confident interval
plt.hist(q2, bins=30)
plt.axvline(q2_lower, color='r', linestyle='dashed', linewidth=2)
plt.axvline(q2_upper, color='r', linestyle='dashed', linewidth=2)

plt.show()
