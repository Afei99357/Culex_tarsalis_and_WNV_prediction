from sklearn import ensemble, metrics, model_selection, preprocessing
import pandas as pd
import numpy as np



# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/CA_human_data_2004_to_2023_final_impute_entire_04_23.csv",
                   index_col=False,
                   header=0)

# drop columns that are not features and drop target
data = data.drop([
    "Date",
    "Year",
    # 'Month',
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

## drop rows if has nan values
data = data.dropna()

## reindex the data
data = data.reset_index(drop=True)

## print 0 variance columns
print(data.columns[data.var() == 0])

## check if any columns has zero variance and drop the columns
data = data.loc[:, data.var() != 0]

# Function to split dataset into k folds
def k_fold_split(data, k):
    fold_size = len(data) // k
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    folds = []
    ## separate the data into k folds
    for i in range(k):
        fold_indices = indices[i * fold_size:(i + 1) * fold_size]
        fold = data.iloc[fold_indices]
        folds.append(fold)

    return folds

cv_folds = 5

cv = k_fold_split(data, k=cv_folds)

## create a list to store the Q2
q2 = []
mse = []

## loop through each fold, use the fold as test set and the rest as training set
for i in range(cv_folds):

    train_data = pd.concat([cv[j] for j in range(cv_folds) if j != i])
    test_data = cv[i].copy()

    ## get the labels
    train_labels = train_data.pop("Human_Disease_Count").values
    test_labels = test_data.pop("Human_Disease_Count").values

    ## random forest
    rf = ensemble.RandomForestRegressor(n_estimators=100, max_depth=50, random_state=42)

    rf.fit(train_data, train_labels)

    y_predict = rf.predict(test_data)

    ## calculate the Q2
    q2.append(metrics.r2_score(test_labels, y_predict))
    ##calculate the MSE
    mse.append(metrics.mean_squared_error(test_labels, y_predict))

print("Cross-validation individual Q2: ", q2)
print("Cross-validation individual MSE: ", mse)

## print average Q2, mse
print("average q2: ", np.mean(q2))
print("average mse: ", np.mean(mse))