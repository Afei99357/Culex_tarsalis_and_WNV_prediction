import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble, metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Load the dataset into a Pandas DataFrame
df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/WNV/california/data/cali_week_impute_monthly_mean_value_with_allele_frequency_RDA_after_drop_high_cor_and_low_target_cor.csv",
    index_col=False,
    header=0)

# Preprocess the DataFrame
df.drop(["Year", 'Month', 'State', "Latitude", "Longitude", 'Land_Area_2010', "Poverty_Estimate_All_Ages",
         "tig00000199_2572223.T", "tig00000379_62942.C", "tig00003801_23907.G"], axis=1, inplace=True)
df_fips = df.pop("FIPS")
df["Date"] = pd.to_datetime(df["Date"])

# Splitting the data
train_rf = df[df['Date'] < pd.to_datetime('2011-01-01')]
test_rf = df[df['Date'] >= pd.to_datetime('2011-01-01')]
original_train_rf_labels = train_rf.pop("Human_WNND_Count").values
test_rf_labels = test_rf.pop("Human_WNND_Count").values
train_rf.drop("Date", axis=1, inplace=True)
test_rf.drop("Date", axis=1, inplace=True)

# Normalizing the Linear Regression features (after separate feature engineering for LR)
# Assume feature engineering for Linear Regression has been done separately
scaler = StandardScaler()
X_train_lr = scaler.fit_transform(train_rf)  # Assuming LR uses the same features as RF for simplicity
X_test_lr = scaler.transform(test_rf)

# Initialize lists to store results
(original_rf_mse, original_rf_r2,
 original_lr_mse, original_lr_r2,
 original_svm_mse, original_svm_r2,
 original_hgbr_mse, original_hgbr_r2,
 original_ensemble_mse, original_ensemble_r2) = ([] for _ in range(10))

(permuted_rf_mse, permuted_rf_r2,
 permuted_lr_mse, permuted_lr_r2,
 permuted_svm_mse, permuted_svm_r2,
 permuted_hgbr_mse, permuted_hgbr_r2,
 permuted_ensemble_mse, permuted_ensemble_r2) = ([] for _ in range(10))


## SVM
clf = SVR(epsilon=.3, gamma=0.002, kernel="rbf", C=100)
clf.fit(X_train_lr, original_train_rf_labels)
y_predict_svm = clf.predict(X_test_lr)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_lr, original_train_rf_labels)  # Simplification for demonstration
y_predict_lr = lr.predict(X_test_lr)



original_lr_mse.append(metrics.mean_squared_error(test_rf_labels, y_predict_lr))
original_lr_r2.append(metrics.r2_score(test_rf_labels, y_predict_lr))
original_svm_mse.append(metrics.mean_squared_error(test_rf_labels, y_predict_svm))
original_svm_r2.append(metrics.r2_score(test_rf_labels, y_predict_svm))



# Execute models 1000 times on original data
for _ in range(1000):
    # Random Forest
    rf = RandomForestRegressor(n_estimators=4, max_depth=3, max_features=0.7, n_jobs=-1)
    rf.fit(train_rf, original_train_rf_labels)
    y_predict_rf = rf.predict(test_rf)

    # Ensemble Predictions
    y_ensemble_pred = 0.5 * y_predict_rf + 0.5 * y_predict_lr

    ## HistGradientBoostingRegressor
    est = ensemble.HistGradientBoostingRegressor(max_iter=1000, max_depth=2, max_leaf_nodes=5, learning_rate=0.1)
    est.fit(train_rf, original_train_rf_labels)
    y_predict_hgbr = est.predict(test_rf)

    # Store Metrics
    original_rf_mse.append(metrics.mean_squared_error(test_rf_labels, y_predict_rf))
    original_rf_r2.append(metrics.r2_score(test_rf_labels, y_predict_rf))
    original_ensemble_mse.append(metrics.mean_squared_error(test_rf_labels, y_ensemble_pred))
    original_ensemble_r2.append(metrics.r2_score(test_rf_labels, y_ensemble_pred))
    original_hgbr_mse.append(metrics.mean_squared_error(test_rf_labels, y_predict_hgbr))
    original_hgbr_r2.append(metrics.r2_score(test_rf_labels, y_predict_hgbr))

# Perform permutation test 1000 times
for _ in range(1000):
    permuted_labels = np.random.permutation(original_train_rf_labels)

    # Random Forest with permuted labels
    rf.fit(train_rf, permuted_labels)
    y_predict_rf_permuted = rf.predict(test_rf)

    # Linear Regression with permuted labels
    lr.fit(X_train_lr, permuted_labels)
    y_predict_lr_permuted = lr.predict(X_test_lr)

    ## SVM with permuted labels
    clf.fit(X_train_lr, permuted_labels)
    y_predict_svm_permuted = clf.predict(X_test_lr)

    # Ensemble Predictions with permuted labels
    y_ensemble_pred_permuted = 0.5 * y_predict_rf_permuted + 0.5 * y_predict_lr_permuted

    ## HistGradientBoostingRegressor with permuted labels
    est.fit(train_rf, permuted_labels)
    y_predict_hgbr_permuted = est.predict(test_rf)

    # Store Permuted Metrics
    permuted_rf_mse.append(metrics.mean_squared_error(test_rf_labels, y_predict_rf_permuted))
    permuted_rf_r2.append(metrics.r2_score(test_rf_labels, y_predict_rf_permuted))
    permuted_lr_mse.append(metrics.mean_squared_error(test_rf_labels, y_predict_lr_permuted))
    permuted_lr_r2.append(metrics.r2_score(test_rf_labels, y_predict_lr_permuted))
    permuted_svm_mse.append(metrics.mean_squared_error(test_rf_labels, y_predict_svm_permuted))
    permuted_svm_r2.append(metrics.r2_score(test_rf_labels, y_predict_svm_permuted))
    permuted_hgbr_mse.append(metrics.mean_squared_error(test_rf_labels, y_predict_hgbr_permuted))
    permuted_hgbr_r2.append(metrics.r2_score(test_rf_labels, y_predict_hgbr_permuted))
    permuted_ensemble_mse.append(metrics.mean_squared_error(test_rf_labels, y_ensemble_pred_permuted))
    permuted_ensemble_r2.append(metrics.r2_score(test_rf_labels, y_ensemble_pred_permuted))

# Function to plot histogram with confidence intervals
def plot_histogram_with_ci(data_list, title, label, save_path):
    plt.figure(figsize=(10, 6))
    plt.hist(data_list, bins=30, alpha=0.5, label=label)
    plt.axvline(np.percentile(data_list, 2.5), color='r', linestyle='dashed', linewidth=1)
    plt.axvline(np.percentile(data_list, 97.5), color='r', linestyle='dashed', linewidth=1)
    plt.legend(loc='upper right')
    plt.title(title)
    plt.savefig(save_path, dpi=300)
    plt.show()

# Plotting the original model performances
plt.figure(figsize=(12, 8))
plt.boxplot([original_ensemble_mse, original_rf_mse, original_lr_mse, original_svm_mse, original_hgbr_mse],
            labels=["Ensemble MSE", "RF MSE", "Linear MSE", "SVM MSE", "HistogramGBRT MSE"])
plt.xticks(rotation=45)
plt.title('Original Model Performances')
plt.tight_layout()
plt.savefig("/Users/ericliao/Desktop/WNV_project_files/WNV/california/plots/original_model_mse_boxplot.png")
plt.show()

# Plotting the original model performances
plt.figure(figsize=(12, 8))
plt.boxplot([original_ensemble_r2, original_rf_r2, original_lr_r2, original_svm_r2, original_hgbr_r2],
            labels=["Ensemble R2", "RF R2", "Linear R2", "SVM R2", "HistogramGBRT R2"])
## add a line for the 0 r2 score
plt.axhline(0, color='red', linestyle='dashed', linewidth=1)
plt.xticks(rotation=45)
plt.title('Original Model Performances')
plt.tight_layout()
plt.savefig("/Users/ericliao/Desktop/WNV_project_files/WNV/california/plots/original_model_r2_boxplot.png", dpi=300)
plt.show()

# Example usage for plotting histograms with 95% CI for ensemble MSE
plot_histogram_with_ci(permuted_ensemble_mse, 'Ensemble MSE Distribution', 'Ensemble MSE', "/Users/ericliao/Desktop/WNV_project_files/WNV/california/plots/permutation_ensemble_mse_distribution.png")
plot_histogram_with_ci(permuted_ensemble_r2, 'Ensemble R2 Distribution', 'Ensemble R2', "/Users/ericliao/Desktop/WNV_project_files/WNV/california/plots/permutation_ensemble_r2_distribution.png")
plot_histogram_with_ci(permuted_rf_mse, 'RF MSE Distribution', 'RF MSE', "/Users/ericliao/Desktop/WNV_project_files/WNV/california/plots/permutation_rf_mse_distribution.png")
plot_histogram_with_ci(permuted_rf_r2, 'RF R2 Distribution', 'RF R2', "/Users/ericliao/Desktop/WNV_project_files/WNV/california/plots/permutation_rf_r2_distribution.png")
plot_histogram_with_ci(permuted_lr_mse, 'Linear MSE Distribution', 'Linear MSE', "/Users/ericliao/Desktop/WNV_project_files/WNV/california/plots/permutation_linear_mse_distribution.png")
plot_histogram_with_ci(permuted_lr_r2, 'Linear R2 Distribution', 'Linear R2', "/Users/ericliao/Desktop/WNV_project_files/WNV/california/plots/permutation_linear_r2_distribution.png")
plot_histogram_with_ci(permuted_svm_mse, 'SVM MSE Distribution', 'SVM MSE', "/Users/ericliao/Desktop/WNV_project_files/WNV/california/plots/permutation_svm_mse_distribution.png")
plot_histogram_with_ci(permuted_svm_r2, 'SVM R2 Distribution', 'SVM R2', "/Users/ericliao/Desktop/WNV_project_files/WNV/california/plots/permutation_svm_r2_distribution.png")
plot_histogram_with_ci(permuted_hgbr_mse, 'HistogramGBRT MSE Distribution', 'HistogramGBRT MSE', "/Users/ericliao/Desktop/WNV_project_files/WNV/california/plots/permutation_hgbr_mse_distribution.png")
plot_histogram_with_ci(permuted_hgbr_r2, 'HistogramGBRT R2 Distribution', 'HistogramGBRT R2', "/Users/ericliao/Desktop/WNV_project_files/WNV/california/plots/permutation_hgbr_r2_distribution.png")
