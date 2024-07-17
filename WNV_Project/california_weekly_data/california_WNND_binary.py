import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import numpy as np

# Load the dataset into a Pandas DataFrame
df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
                 "add_0_for_no_wnv/cali_week_wnnd_human_all_features.csv", index_col=False)

# add one column to store the binary value of Human_WNND_Count
df["Human_WNND_Count_Binary"] = df["Human_WNND_Count"].apply(lambda x: 1 if x > 0 else 0)

# drop columns that are not features and drop target
df = df.drop(["County", "State",
              "FIPS",
              "Year", 'Month',
              "County_Seat_Latitude",
              "County_Seat_Longitude",
              "Poverty_Estimate_All_Ages",
              "Human_WNND_Count"
              ], axis=1)

### train and test data ####
# convert "Date" column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Split the data into training and testing sets
train_rf = df[df['Date'] < '2011-01-01']
test_rf = df[(df['Date'] >= '2011-01-01')]

# Get labels
train_rf_labels = train_rf.pop("Human_WNND_Count_Binary").values
test_rf_labels = test_rf.pop("Human_WNND_Count_Binary").values

train_rf.pop("Date")
test_rf.pop("Date")

# get the column names
train_column_names = train_rf.columns
test_column_names = test_rf.columns

######################### RF ######################################
# Create a random forest classifier
rf_class = RandomForestClassifier(n_estimators=1000, bootstrap=True, oob_score=True, random_state=42)

# Train the classifier
rf_class.fit(train_rf, train_rf_labels)

# Get the feature importances
importances = rf_class.feature_importances_

# Get the predictions
predictions = rf_class.predict(test_rf)

# Get the f1 score
f1_score = f1_score(test_rf_labels, predictions)

# Get the roc_auc score
roc_auc_score = roc_auc_score(test_rf_labels, predictions)

# Get the confusion matrix
confusion_matrix = confusion_matrix(test_rf_labels, predictions)

# Calculate the Q2 score
# Calculate the sum of squared errors of prediction (SSE_pred)
SSE_pred = np.sum((test_rf_labels - predictions) ** 2)

# Calculate the sum of squared errors of the mean (SSE_mean)
SSE_mean = np.sum((test_rf_labels - np.mean(test_rf_labels)) ** 2)

# Calculate the Q2 score
q2_score = 1 - (SSE_pred / SSE_mean)

# Print the results
print("F1 Score: ", f1_score)
print("Q2 Score: ", q2_score)
print("ROC_AUC Score: ", roc_auc_score)

# importance of each feature
feature_importance = pd.DataFrame(list(zip(train_column_names, importances)), columns=["Feature", "Importance"])
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)
print(feature_importance)

