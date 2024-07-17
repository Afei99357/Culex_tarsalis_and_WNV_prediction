from sklearn import ensemble, metrics
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/"
                   "human/cdc_human_1999_to_2023/WNV_human_and_non_human_yearly_climate_demographic_bird.csv", index_col=0)

## remove any space and comma in Population column
data["Population"] = data["Population"].str.replace(",", "").str.strip()

## convert Population column to numeric
data["Population"] = pd.to_numeric(data["Population"], errors='coerce')

# select the columns after column Date as predictors
date_index = data.columns.get_loc("Date")

# Select columns after the "Date" column as predictors
data_pred = data.iloc[:, date_index+1:]

# get Year column from data and add it to data_pred
data_pred['Year'] = data['Year']

## add target column
data_pred["Neuroinvasive_disease_cases"] = data["Neuroinvasive_disease_cases"]

# drop nan values
data_pred = data_pred.dropna()

## reset index
data_pred = data_pred.reset_index(drop=True)

### train and test data ####
train = data_pred[(data_pred["Year"] < 2018)]
test = data_pred[(data_pred["Year"] >= 2018)]

# Get labels
train_labels = train.pop("Neuroinvasive_disease_cases").values
test_labels = test.pop("Neuroinvasive_disease_cases").values

## remove time column
train.pop("Year")
test.pop("Year")

# get the column names
train_column_names = train.columns
test_column_names = test.columns

######################### RF ######################################
## Random Forest Classifier
rf = ensemble.RandomForestRegressor(n_estimators=4, max_depth=3, max_features=0.7, n_jobs=-1)

rf.fit(train, train_labels)

y_predict = rf.predict(test)

# Classify risk levels based on predicted values
# Define thresholds for risk levels (e.g., low, moderate, high)
threshold_low = 5  # Example threshold for low risk
threshold_high = 50  # Example threshold for high risk

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

### evaluate the model ####

## acuracy
accuracy = metrics.accuracy_score(actual_risk_levels, risk_levels)

## confusion matrix
conf_matrix = metrics.confusion_matrix(actual_risk_levels, risk_levels)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate classification report
class_report = metrics.classification_report(actual_risk_levels, risk_levels)

# Print classification report
print("\nClassification Report:")
print(class_report)




