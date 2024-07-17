from sklearn import metrics
from sklearn.svm import SVR
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble, metrics


# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/Europe_data/adding_0_case/"
                   "europe_data_with_coordinates_landuse_climate_0_case.csv", index_col=0)


# select the columns as predictors
data_pred = data.iloc[:, 11:]

# convert "Time" column contains year value to datetime
data_pred["Time"] = pd.to_datetime(data["Time"], format="%Y")
data_pred["NumValue"] = data["NumValue"]

# drop nan values
data_pred = data_pred.dropna()

### 111 ####
### train and test data ####
train = data_pred[(data_pred["Time"] < "2021")]
test = data_pred[(data_pred["Time"] >= "2021")]

# Get labels
train_labels = train.pop("NumValue").values
test_labels = test.pop("NumValue").values

## remove time column
train.pop("Time")
test.pop("Time")

# get the column names
train_column_names = train.columns
test_column_names = test.columns
# normalize the data
scaler = StandardScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)

######################### SVM ######################################

clf = SVR(epsilon=.3, gamma=0.002, kernel="rbf", C=100)

clf.fit(train, train_labels)

y_predict = clf.predict(test)

# store test labels and predicted labels in a dataframe
df = pd.DataFrame({"test_labels": test_labels, "y_predict": y_predict})
df.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/Europe_data/adding_0_case/SVM_euro_0_case.csv")

# ## get the mse, r2 score
mse = metrics.mean_squared_error(test_labels, y_predict)
r2 = metrics.r2_score(test_labels, y_predict)
# for comparison, get the mse and r2 score of a fake model that always predict the mean of the target
fake_model_mse = metrics.mean_squared_error(test_labels, [train_labels.mean()] * len(test_labels))
fake_model_r2 = metrics.r2_score(test_labels, [train_labels.mean()] * len(test_labels))

# #print the mse and r2 score
print("The mean squared error of SVM Model verse Fake Model: {:.03}, vs {:.03}".format(mse, fake_model_mse))
print("The r2 score of SVM Model verse Fake Model: {:.03}, vs {:.03}".format(r2, fake_model_r2))

####################################################################