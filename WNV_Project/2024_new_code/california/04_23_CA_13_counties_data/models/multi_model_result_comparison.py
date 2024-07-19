import pandas as pd
from matplotlib import pyplot as plt

## load SVM results
svm_results = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/result/plots/"
                          "SVM/hyperparameter_tuning_plots/hyperparameter_tuning_svm_impute_0_q2_rmse.csv")

## load RF results
rf_results = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/result/plots/"
                         "RF/hyperparameter_tuning_plots/hyperparameter_tuning_rf_impute_0_q2_rmse.csv")

## load HGBR results
hgbr_results = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/result/plots/"
                           "HGBR/hgbr_impute_0_tuning_year_q2_rmse.csv")

## plot the Q^2 and RMSE for SVM, RF and HGBR
plt.figure(figsize=(10, 5))

plt.plot(svm_results["tuning_year"], svm_results["q2"], label="SVM", color="red")
plt.plot(rf_results["tuning_year"], rf_results["q2"], label="RF", color="blue")
plt.plot(hgbr_results["tuning_year"], hgbr_results["q2"], label="HGBR", color="green")

plt.xlabel("Tuning year")
plt.ylabel("Q^2")
plt.title("Q^2 comparison between SVM, RF and HGBR")

## keep the x axis in integer
plt.xticks(svm_results["tuning_year"].astype(int))

plt.legend(loc="upper left")
plt.savefig("/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/result/plots/multi_models_Q2_comparison.png", dpi=300)

plt.show()

## plot the RMSE
plt.figure(figsize=(10, 5))

plt.plot(svm_results["tuning_year"], svm_results["RMSE"], label="SVM", color="red")
plt.plot(rf_results["tuning_year"], rf_results["RMSE"], label="RF", color="blue")
plt.plot(hgbr_results["tuning_year"], hgbr_results["RMSE"], label="HGBR", color="green")

plt.xlabel("Tuning year")
plt.ylabel("RMSE")
plt.title("RMSE comparison between SVM, RF and HGBR")

## keep the x axis in integer
plt.xticks(svm_results["tuning_year"].astype(int))

plt.legend(loc="upper left")
plt.savefig("/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/result/plots/multi_models_RMSE_comparison.png", dpi=300)

plt.show()
