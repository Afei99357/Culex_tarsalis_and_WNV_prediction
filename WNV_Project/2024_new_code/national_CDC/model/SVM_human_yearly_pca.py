#### run PCA on the data #####
#
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn import metrics


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

## preprocess the data
scaler = StandardScaler()

#create copy of DataFrame
scaled_data = data_pred.copy()

#created scaled version of DataFrame
scaled_df = pd.DataFrame(scaler.fit_transform(scaled_data), columns=scaled_data.columns)

## drop the target column
scaled_df = scaled_df.drop("Neuroinvasive_disease_cases", axis=1)

column_list = scaled_df.columns.values.tolist()

## run PCA on the data
pca = PCA()

## fit the data
pca.fit(scaled_df)

## total variance explained by the first 10 principal components
print(pca.explained_variance_ratio_[:10].sum())


## store the first 10 principal components as a new predictor
# data_pred_pca = pca.transform(scaled_df)[:, :10]
## add the target column to the data_pred_pca
# data_pred_pca = pd.DataFrame(data_pred_pca, columns=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10"])
# data_pred_pca["Neuroinvasive_disease_cases"] = data_pred["Neuroinvasive_disease_cases"]

## store all principal components as a new predictor
data_pred_pca = pca.transform(scaled_df)[:, :]
## convert the data_pred_pca to a dataframe and feed all PCs to the model
data_pred_pca = pd.DataFrame(data_pred_pca, columns=[f"PC{i}" for i in range(1, data_pred_pca.shape[1] + 1)])
data_pred_pca["Neuroinvasive_disease_cases"] = data_pred["Neuroinvasive_disease_cases"]

## add Year column to data_pred_pca
data_pred_pca["Year"] = data_pred["Year"]

### train and test data ####
train = data_pred_pca[(data_pred_pca["Year"] < 2018)]
test = data_pred_pca[(data_pred_pca["Year"] >= 2018)]

## get labels
train_labels = train.pop("Neuroinvasive_disease_cases").values
test_labels = test.pop("Neuroinvasive_disease_cases").values

## remove time column
train.pop("Year")
test.pop("Year")

## FIT SVM model
clf = SVR(epsilon=.3, gamma=0.002, kernel="rbf", C=100)

clf.fit(train, train_labels)

y_predict = clf.predict(test)

# store test labels and predicted labels in a dataframe
df = pd.DataFrame({"test_labels": test_labels, "y_predict": y_predict})
df.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/result/SVM/human_yearly_cdc_svm_pca.csv")

## get the mse, r2 score
mse = metrics.mean_squared_error(test_labels, y_predict)
r2 = metrics.r2_score(test_labels, y_predict)
# for comparison, get the mse and r2 score of a fake model that always predict the mean of the target
fake_model_mse = metrics.mean_squared_error(test_labels, [train_labels.mean()] * len(test_labels))
fake_model_r2 = metrics.r2_score(test_labels, [train_labels.mean()] * len(test_labels))
print(f"Mean squared error: {mse}")
print(f"R2 score: {r2}")
print(f"Fake model mean squared error: {fake_model_mse}")
print(f"Fake model r2 score: {fake_model_r2}")



