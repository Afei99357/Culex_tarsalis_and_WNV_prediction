from sklearn.svm import SVC
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import numpy as np
import seaborn as sns
from sklearn import metrics


# Load the dataset into a Pandas DataFrame
df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/"
                   "human/cdc_human_1999_to_2023/WNV_human_and_non_human_yearly_climate_demographic_bird.csv", index_col=None)

state = "michigan"
state_code = 'MI'

df = df[df["State"] == state]

## adding a empty column Binary_Target
df["Binary_Target"] = 0

## if any column in columns Neuroinvasive_disease_cases, Identified_by_Blood_Donor_Screening, Total_Bird_WNV_Count, Mos_WNV_Count and Horse_WNV_Count has value greater than 0, then set Binary_Target to 1.
df.loc[(df["Neuroinvasive_disease_cases"] > 0) | (df["Identified_by_Blood_Donor_Screening"] > 0) | (df["Total_Bird_WNV_Count"] > 0) | (df["Mos_WNV_Count"] > 0) | (df["Horse_WNV_Count"] > 0), "Binary_Target"] = 1
## any binary target is nan, set it to 0
df["Binary_Target"] = df["Binary_Target"].fillna(0)

# ## output binary dataset
# df.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/"
#                    "human/cdc_human_1999_to_2023/WNV_human_and_non_human_yearly_climate_demographic_bird_binary.csv", index=False)

## check nan values in Binary_Target
print(df["Binary_Target"].isnull().sum())

## remove any space and comma in Population column
df["Population"] = df["Population"].str.replace(",", "").str.strip()

# convert population to float
df["Population"] = df["Population"].astype(float)

# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/"
#                    "human/cdc_human_1999_to_2023/WNV_human_and_non_human_yearly_climate_demographic_bird_binary.csv", index_col=False)


# select the columns after column Date as predictors
date_index = df.columns.get_loc("Date")

# Select columns after the "Date" column as predictors
data_pred = df.iloc[:, date_index+1:]

# get Year column from data and add it to data_pred
data_pred['Year'] = df['Year']

## add target column
data_pred["Binary_Target"] = df.pop("Binary_Target")

# drop nan values
data_pred = data_pred.dropna()

## reset index
data_pred = data_pred.reset_index(drop=True)

### train and test data ####
train = data_pred[(data_pred["Year"] < 2018)]
test = data_pred[(data_pred["Year"] >= 2018)]

# Get labels
train_labels = train.pop("Binary_Target").values
test_labels = test.pop("Binary_Target").values

## remove time column
train.pop("Year")
test.pop("Year")

# get the column names
train_column_names = train.columns
test_column_names = test.columns

# get the dataset before normalized
df_train_preprocessed = train
df_test_preprocessed = test

# normalize the data
scaler = StandardScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)

# run SVM model
# create a SVM model
clf = SVC(kernel="rbf", C=1, gamma='auto', probability=True)

# fit the model
clf.fit(train, train_labels)

# predict the target
y_pred_proba = clf.predict_proba(test)

y_pred_class_1 = y_pred_proba[:, 1]

## found how many y_pred is equal to 0 and how many y_pred is greater than 0
print("y_pred equal to 0: ", np.sum(y_pred_class_1 == 0))
print("y_pred greater than 0: ", np.sum(y_pred_class_1 > 0))

# calculate the accuracy of the model and F1 score

# decide the threshold to maximize the F1 score
threshold = np.arange(0, 1, 0.01)
f1_scores = []
for i in threshold:
    y_pred_binary = np.where(y_pred_class_1 > i, 1, 0)
    tp = np.sum((y_pred_binary == 1) & (test_labels == 1))
    # if tp is 0, continue to the next iteration
    if tp == 0:
        continue
    fp = np.sum((y_pred_binary == 1) & (test_labels == 0))
    fn = np.sum((y_pred_binary == 0) & (test_labels == 1))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1_scores.append(2 * precision * recall / (precision + recall))
    print("Threshold: ", i, "F1 score: ", 2 * precision * recall / (precision + recall))
print("The threshold that maximizes the F1 score is: ", threshold[np.argmax(f1_scores)])

# the threshold that maximum the f1 score which is not nan
threshold = threshold[np.argmax(f1_scores)]


# convert the predicted probability to binary value
y_pred_binary = np.where(y_pred_class_1 > threshold, 1, 0)
# calculate the accuracy
accuracy = np.sum(y_pred_binary == test_labels) / len(test_labels)
print("Accuracy: ", accuracy)
# calculate the F1 score
tp = np.sum((y_pred_binary == 1) & (test_labels == 1))
fp = np.sum((y_pred_binary == 1) & (test_labels == 0))
fn = np.sum((y_pred_binary == 0) & (test_labels == 1))
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * precision * recall / (precision + recall)
print("F1 score: ", f1_score)

# plot Receiver Operating Characteristic (ROC) Curve
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic')
fpr, tpr, _ = roc_curve(test_labels, y_pred_class_1)
roc_auc = auc(fpr, tpr)
ax.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
ax.legend(loc="lower right")
# plt.show()
plt.savefig("/Users/ericliao/Desktop/SVM_binary_ROC_" + state_code + ".png")
plt.close()


# Plotting the probability distribution of the predicted probabilities can provide insights into the separation between the two classes.
# plot the histogram of the predicted probability
fig, ax = plt.subplots()
plt.hist(y_pred_class_1)
plt.xlabel("Predicted probability of WNV")
plt.ylabel("Frequency")
plt.title("Histogram of predicted probability")
plt.savefig("/Users/ericliao/Desktop/SVM_binary_histogram_" + state_code + ".png")
plt.close()

# plot predicted probability vs actual probability
fig, ax = plt.subplots()
plt.scatter(y_pred_class_1, test_labels)
plt.xlabel("Predicted probability of WNV")
plt.ylabel("Actual probability of WNV")
plt.title("Predicted probability vs actual probability")
plt.savefig("/Users/ericliao/Desktop/SVM_binary_predicted_probability_vs_actual_probability_" + state_code + ".png")
plt.close()

# fringe plot of predicted probability vs actual probability
# define a range of values for the predicted probability
bins = np.linspace(0, 1, 10)
# create a new column in the dataframe that contains the predicted probability
df_test_preprocessed["Predicted"] = y_pred_class_1
# create a new column in the dataframe that contains the actual probability
df_test_preprocessed["Actual"] = test_labels
# group the dataframe by the predicted probability
groups = df_test_preprocessed.groupby(pd.cut(df_test_preprocessed["Predicted"], bins))
# calculate the mean of the actual probability for each group
actual_prob = groups["Actual"].mean()
# calculate the mean of the predicted probability for each group
predicted_prob = groups["Predicted"].mean()
# plot the fringe plot
fig, ax = plt.subplots()
# plt.plot(predicted_prob, actual_prob, marker='o')
plt.scatter(df_test_preprocessed["Predicted"], df_test_preprocessed["Actual"], alpha=0.5)
plt.xlabel("Predicted probability of WNV")
plt.ylabel("Actual probability of WNV")
plt.title("Scatter plot of predicted probability vs actual probability")
plt.savefig("/Users/ericliao/Desktop/SVM_binary_scatter_" + state_code + ".png")
plt.close()

# KDE plot for predicted and actual probability
fig, ax = plt.subplots()
sns.kdeplot(df_test_preprocessed["Predicted"], shade=True, label="Predicted probability")
sns.kdeplot(df_test_preprocessed["Actual"], shade=True, label="Actual probability")
plt.xlabel("Probability of WNV")
plt.ylabel("Density")
plt.legend()
plt.title("KDE plot of predicted probability vs actual probability")
plt.savefig("/Users/ericliao/Desktop/SVM_binary_KDE_" + state_code + ".png")
plt.close()





