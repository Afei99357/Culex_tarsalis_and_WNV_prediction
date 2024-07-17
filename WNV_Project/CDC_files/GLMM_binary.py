import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import numpy as np
import seaborn as sns


# load data
df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/cdc_sum_organism_all_binary.csv",
                 index_col=0)

# remove all the comma in the data
df["Population"] = df["Population"].str.replace(",", "")
# convert population to float
df["Population"] = df["Population"].astype(float)

# data cleaning
# drop the rows where the value is nan and reset the index
df = df.dropna()

# get training and testing data, use data before 2017 as training data and after 2017 as testing data
df_train = df[df["Year"] < 2017]
df_test = df[df["Year"] >= 2017]

# reset the index for train and test data
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# drop the columns that are not needed
df_train = df_train.drop(columns=["Year", "Month", "FIPS", "County", "State", "County_Seat", "Total_Organism_WNV_Count",
                                  "Date"])

df_test = df_test.drop(columns=["Year", "Month", "FIPS", "County", "State", "County_Seat", "Total_Organism_WNV_Count",
                                "Date"])

# get the dataset without longitude and latitude
df_train_preprocessed = df_train.drop(columns=['Longitude', 'Latitude', 'Binary_Target'])
df_test_preprocessed = df_test.drop(columns=['Longitude', 'Latitude', 'Binary_Target'])

# preprocessing the df_train_preprocessed and df_test_preprocessed
scaler = StandardScaler()
scaler.fit(df_train_preprocessed)
df_train_preprocessed = scaler.transform(df_train_preprocessed)
df_test_preprocessed = scaler.transform(df_test_preprocessed)

# convert the numpy array to dataframe
df_train_preprocessed = pd.DataFrame(df_train_preprocessed,
                                     columns=df_train.drop(columns=['Longitude', 'Latitude', 'Binary_Target']).columns)
df_test_preprocessed = pd.DataFrame(df_test_preprocessed,
                                    columns=df_test.drop(columns=['Longitude', 'Latitude', 'Binary_Target']).columns)

# adding the columns of longitude, latitude and Bianry_Target back to df_train_preprocessed and df_test_preprocessed
df_train_preprocessed["Longitude"] = df_train["Longitude"]
df_train_preprocessed["Latitude"] = df_train["Latitude"]
df_train_preprocessed["Binary_Target"] = df_train["Binary_Target"]
df_test_preprocessed["Longitude"] = df_test["Longitude"]
df_test_preprocessed["Latitude"] = df_test["Latitude"]
df_test_preprocessed["Binary_Target"] = df_test["Binary_Target"]


# create formula where only Longitude and Latitude are random effect, GLMM model
formula = "Binary_Target ~ Population + u10_1m_shift + v10_1m_shift + t2m_1m_shift + lai_hv_1m_shift + lai_lv_1m_shift " \
          "+ src_1m_shift + sf_1m_shift + sro_1m_shift + tp_1m_shift + Northern_Mockingbird + House_Finch " \
          "+ Canada_Goose + Brown_Thrasher + Common_Grackle + House_Sparrow + American_Goldfinch + American_Robin " \
          "+ Blue_Jay + Gray_Catbird + American_Crow + Northern_Cardinal + Song_Sparrow + Green_Jay " \
          "+ White_winged_Dove + Ring_billed_Gull + Longitude + Latitude"

# fit the GLMM model
model = smf.glm(formula=formula, data=df_train_preprocessed, family=sm.families.Binomial()).fit()

# print the summary of the model
print(model.summary())

# testing actual target
test_origin_target = df_test_preprocessed.pop("Binary_Target")

# predict the probability of the testing data
y_pred = model.predict(df_test_preprocessed)

# print the predicted probability along with the actual value
print(pd.DataFrame({"Actual": test_origin_target, "Predicted": y_pred}))

# output the predicted probability along with the actual value to csv file
pd.DataFrame({"Actual": test_origin_target, "Predicted": y_pred}).to_csv("/Users/ericliao/Desktop/GLMM_binary_result.csv")

# calculate the accuracy of the model and F1 score

# decide the threshold to maximize the F1 score
threshold = np.arange(0, 1, 0.01)
f1_scores = []
for i in threshold:
    y_pred_binary = np.where(y_pred > i, 1, 0)
    tp = np.sum((y_pred_binary == 1) & (test_origin_target == 1))
    fp = np.sum((y_pred_binary == 1) & (test_origin_target == 0))
    fn = np.sum((y_pred_binary == 0) & (test_origin_target == 1))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_scores.append(2 * precision * recall / (precision + recall))
print("The threshold that maximizes the F1 score is: ", threshold[np.argmax(f1_scores)])

# the threshold that maximum the f1 score
threshold = threshold[np.argmax(f1_scores)]

# convert the predicted probability to binary value
y_pred_binary = np.where(y_pred > threshold, 1, 0)
# calculate the accuracy
accuracy = np.sum(y_pred_binary == test_origin_target) / len(test_origin_target)
print("Accuracy: ", accuracy)
# calculate the F1 score
tp = np.sum((y_pred_binary == 1) & (test_origin_target == 1))
fp = np.sum((y_pred_binary == 1) & (test_origin_target == 0))
fn = np.sum((y_pred_binary == 0) & (test_origin_target == 1))
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * precision * recall / (precision + recall)
print("F1 score: ", f1_score)

# output y_pred_binary and test_origin_target to csv file
pd.DataFrame({"Actual": test_origin_target, "Predicted": y_pred_binary}).to_csv("/Users/ericliao/Desktop/"
                                                                                "GLMM_maximum_f1_binary_prediction_vs_real.csv")

# plot Receiver Operating Characteristic (ROC) Curve
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic')
fpr, tpr, _ = roc_curve(test_origin_target, y_pred)
roc_auc = auc(fpr, tpr)
ax.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
ax.legend(loc="lower right")
# plt.show()
plt.savefig("/Users/ericliao/Desktop/GLMM_binary_ROC.png")
plt.close()


# Plotting the probability distribution of the predicted probabilities can provide insights into the separation between the two classes.
# plot the histogram of the predicted probability
fig, ax = plt.subplots()
plt.hist(y_pred)
plt.xlabel("Predicted probability of WNV")
plt.ylabel("Frequency")
plt.title("Histogram of predicted probability")
plt.savefig("/Users/ericliao/Desktop/GLMM_binary_histogram.png")
plt.close()

# plot predicted probability vs actual probability
fig, ax = plt.subplots()
plt.scatter(y_pred, test_origin_target)
plt.xlabel("Predicted probability of WNV")
plt.ylabel("Actual probability of WNV")
plt.title("Predicted probability vs actual probability")
plt.savefig("/Users/ericliao/Desktop/GLMM_binary_scatter.png")
plt.close()

# fringe plot of predicted probability vs actual probability
# define a range of values for the predicted probability
bins = np.linspace(0, 1, 10)
# create a new column in the dataframe that contains the predicted probability
df_test_preprocessed["Predicted"] = y_pred
# create a new column in the dataframe that contains the actual probability
df_test_preprocessed["Actual"] = test_origin_target
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
plt.savefig("/Users/ericliao/Desktop/GLMM_binary_scatter.png")
plt.close()

# KDE plot for predicted and actual probability
fig, ax = plt.subplots()
sns.kdeplot(df_test_preprocessed["Predicted"], shade=True, label="Predicted probability")
sns.kdeplot(df_test_preprocessed["Actual"], shade=True, label="Actual probability")
plt.xlabel("Probability of WNV")
plt.ylabel("Density")
plt.legend()
plt.title("KDE plot of predicted probability vs actual probability")
plt.savefig("/Users/ericliao/Desktop/GLMM_binary_KDE.png")
plt.close()





