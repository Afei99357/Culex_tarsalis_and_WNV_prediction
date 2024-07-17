import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import ensemble, metrics
from sklearn.linear_model import LinearRegression


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

# ## store the first 10 principal components as a new predictor
# data_pred_pca = pca.transform(scaled_df)[:, :10]
# ## add the target column to the data_pred_pca
# data_pred_pca = pd.DataFrame(data_pred_pca, columns=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10"])
# data_pred_pca["Neuroinvasive_disease_cases"] = data_pred["Neuroinvasive_disease_cases"]

# ## store the first 10 principal components as a new predictor
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

## RUN THE MODEL 100 times and record the mse and r2 score
rf_mse_list = []
rf_r2_list = []
est_mse_list = []
est_r2_list = []
for i in range(100):
    ######################### RF ######################################
    ## Random Forest Classifier
    rf = ensemble.RandomForestRegressor(n_estimators=4, max_depth=3, max_features=0.7, n_jobs=-1)

    rf.fit(train, train_labels)

    y_predict = rf.predict(test)

    ## HistGradientBoostingRegressor
    est = ensemble.HistGradientBoostingRegressor(max_iter=1000, max_depth=2, max_leaf_nodes=5, learning_rate=0.1)

    est.fit(train, train_labels)

    y_predict_est = est.predict(test)

    # store test labels and predicted labels in a dataframe
    df = pd.DataFrame({"test_labels": test_labels, "y_predict": y_predict})
    df.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/result/RF/human_yearly_cdc_rf.csv")

    df_est = pd.DataFrame({"test_labels": test_labels, "y_predict": y_predict})
    df_est.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/result/HistogramGBRT/human_yearly_cdc_est.csv")

    # ## get the mse, r2 score
    mse = metrics.mean_squared_error(test_labels, y_predict)
    r2 = metrics.r2_score(test_labels, y_predict)

    mse_est = metrics.mean_squared_error(test_labels, y_predict_est)
    r2_est = metrics.r2_score(test_labels, y_predict_est)

    rf_mse_list.append(mse)
    rf_r2_list.append(r2)
    est_mse_list.append(mse_est)
    est_r2_list.append(r2_est)

## print average mse and r2 score
print("average mse: ", sum(rf_mse_list)/len(rf_mse_list))
print("average r2: ", sum(rf_r2_list)/len(rf_r2_list))
print("HistogramGBRT average mse: ", sum(est_mse_list)/len(est_mse_list))
print("HistogramGBRT average r2: ", sum(est_r2_list)/len(est_r2_list))


## run the linear regression model and record the mse and r2 score
linear_regression = LinearRegression()
linear_regression.fit(train, train_labels)

y_predict = linear_regression.predict(test)

# calculate the mse and r2 score
lr_mse = metrics.mean_squared_error(test_labels, y_predict)

lr_r2 = metrics.r2_score(test_labels, y_predict)

## print the mse and r2 score
print("lr_mse: ", lr_mse)
print("lr_r2: ", lr_r2)

### plot boxplot of mse and r2 score
import matplotlib.pyplot as plt

plt.boxplot([rf_mse_list, rf_r2_list, [lr_mse], [lr_r2], est_mse_list, est_r2_list], labels=["rf_mse", "rf_r2", "lr_mse", "lr_r2", "est_mse", "est_r2"])
plt.show()

