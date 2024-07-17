from sklearn import ensemble, metrics
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/data/cali_week_impute_monthly_mean_value_with_allele_frequency_RDA_after_drop_high_cor_and_low_target_cor.csv",
                   index_col=False,
                   header=0)

# Drop columns that are not features and drop target
data = data.drop([
    "Year",
    'Month',
    'FIPS',
    'State',
    "Latitude",
    "Longitude",
    'Land_Area_2010',
    "Poverty_Estimate_All_Ages",
    "tig00000199_2572223.T",
    "tig00000379_62942.C",
    "tig00003801_23907.G"
], axis=1)

data = data.dropna()

# Convert "Date" column to datetime
data["Date"] = pd.to_datetime(data["Date"])

### Train and test data ####
train = data[data['Date'] < '2010-01-01']
test = data[(data['Date'] >= '2010-01-01')]

# Get labels
train_labels = train.pop("Human_WNND_Count").values
test_labels = test.pop("Human_WNND_Count").values

train.pop("Date")
test.pop("Date")

# Get the column names
train_column_names = train.columns
test_column_names = test.columns

# Run the RF model to predict the number of WNV cases
rf = ensemble.RandomForestRegressor(n_estimators=4, max_depth=3, max_features=0.7, n_jobs=-1)
rf.fit(train, train_labels)
y_predict = rf.predict(test)

# Classify risk levels based on predicted values
# Define thresholds for risk levels (e.g., low, moderate, high)
threshold_low = 3  # Example threshold for low risk
threshold_high = 10  # Example threshold for high risk

# Classify regions or populations based on predicted WNV cases
risk_levels = []
for prediction in y_predict:
    if prediction < 1:
        risk_levels.append("No Risk")
    elif prediction >= 1 and prediction < threshold_low:
        risk_levels.append("Low Risk")
    elif prediction >= threshold_low and prediction < threshold_high:
        risk_levels.append("Moderate Risk")
    else:
        risk_levels.append("High Risk")


# Define the actual risk levels based on the number of cases in the test data
actual_risk_levels = []
for label in test_labels:
    if label < 1:
        actual_risk_levels.append("No Risk")
    elif label >= 1 and label < threshold_low:
        actual_risk_levels.append("Low Risk")
    elif label >= threshold_low and label < threshold_high:
        actual_risk_levels.append("Moderate Risk")
    else:
        actual_risk_levels.append("High Risk")

# Calculate confusion matrix
conf_matrix = confusion_matrix(actual_risk_levels, risk_levels)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Calculate classification report
class_report = classification_report(actual_risk_levels, risk_levels)

# Print classification report
print("\nClassification Report:")
print(class_report)
