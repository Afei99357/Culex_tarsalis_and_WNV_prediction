from sklearn import metrics
from sklearn.svm import SVR
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
                   "add_0_for_no_wnv/cali_week_wnnd_multi_years_all_features_ebirds.csv", index_col=0)

# drop columns that are not features and drop target
data = data.drop(["State", "County", "Year", 'Month', "County_Seat_Latitude", "County_Seat_Longitude", "FIPS",
                  "Human_WNND_Count",
                  # "Human_WNND_Rate"
                  "Population"
                  ], axis=1)


data = data.dropna()

# convert "Date" column to datetime
data["Date"] = pd.to_datetime(data["Date"])

# check the standard deviation of the target
print("The standard deviation of the target is: ", data["Human_WNND_Rate"].std())

### train and test data ####
train = data[(data["Date"] > "2004-01-01") & (data["Date"] < "2011-01-01")]
test = data[(data["Date"] >= "2011-01-01") & (data["Date"] < "2011-12-31")]

# Get labels
train_labels = train.pop("Human_WNND_Rate").values
test_labels = test.pop("Human_WNND_Rate").values

train.pop("Date")
test.pop("Date")

# get the column names
train_column_names = train.columns
test_column_names = test.columns

# normalize the data
scaler = StandardScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)
######################### SVM ######################################

clf = SVR(C=11, epsilon=.3, gamma=0.002)

clf.fit(train, train_labels)

y_predict = clf.predict(test)

# store test labels and predicted labels in a dataframe
df = pd.DataFrame({"test_labels": test_labels, "y_predict": y_predict})
df.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/"
          "results/add_0_for_no_wnv/SVM_cali_multi_Years_result.csv")

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