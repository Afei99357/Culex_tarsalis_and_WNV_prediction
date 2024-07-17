import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble, metrics
from scipy.stats import spearmanr


# Load the dataset into a Pandas DataFrame
df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
                 "add_0_for_no_wnv/cali_week_wnnd_multi_years_all_features_mosquitoes.csv", index_col=0)

# drop columns that are not features and drop target
df = df.drop(["County",
              "State",
              # "Year", 'Month',
              "County_Seat_Latitude",
              "County_Seat_Longitude",
              "Poverty_Estimate_All_Ages",
              "Human_WNND_Rate",
              "American Goldfinch",
              "House Finch",
              "House Sparrow",
              "American Crow",
              "American Robin",
              "WNV_Mos_Count"
              ], axis=1)

# drop any columns contains _2m_
df = df[df.columns.drop(list(df.filter(regex='_2m_')))]

# df = df.dropna()

# adding a new column to store the Rank_WNV_WNND, during each year and month,
# get the the rank of Human_WNND_Count for each county
df["Rank_WNV_WNND"] = df.groupby(["Year", "Month"])["Human_WNND_Count"].rank("dense", ascending=False)

# drop the Human_WNND_Count column, Year and Month columns
df = df.drop(["Human_WNND_Count", "Year", "Month"], axis=1)

# # Only choose the Year, Month, Human_WNND_Count, and Rank_WNV_WNND columns
# df_test = df[["Year", "Month", "Human_WNND_Count", "Rank_WNV_WNND", 'FIPS']]
#
# # order first by Year, then by Month
# df_test = df_test.sort_values(by=["Year", "Month"])

### train and test data ####
# convert "Date" column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Split the data into training and testing sets
train_rf = df[df['Date'] < '2011-01-01']
test_rf = df[(df['Date'] >= '2011-01-01')]

# Get labels
train_rf_labels = train_rf.pop("Rank_WNV_WNND").values
test_rf_labels = test_rf.pop("Rank_WNV_WNND").values

train_rf.pop("Date")
test_rf.pop("Date")

# get the column names
train_column_names = train_rf.columns
test_column_names = test_rf.columns

######################### RF ######################################
## Random Forest Classifier for ordinal classification##
rf = ensemble.RandomForestClassifier(n_estimators=4, max_depth=3, max_features=0.7, n_jobs=-1)
rf.fit(train_rf, train_rf_labels)

# predict the test data
y_predict_rf = rf.predict(test_rf)

# create a dataframe to store prediction results and actual results
df_rf = pd.DataFrame({'Actual': test_rf_labels, 'Predicted': y_predict_rf})

# get the accuracy score
accuracy_rf = metrics.accuracy_score(test_rf_labels, y_predict_rf)

# get the confusion matrix
confusion_matrix_rf = metrics.confusion_matrix(test_rf_labels, y_predict_rf)

# get the classification report
classification_report_rf = metrics.classification_report(test_rf_labels, y_predict_rf)

# Calculate Spearman correlation rank
correlation, p_value = spearmanr(test_rf_labels, y_predict_rf)

# print out the results
print("Random Forest Classifier")
print("Accuracy:", accuracy_rf)
print("Confusion Matrix:\n", confusion_matrix_rf)
print("Classification Report:\n", classification_report_rf)
print("Spearman correlation rank:", correlation)


