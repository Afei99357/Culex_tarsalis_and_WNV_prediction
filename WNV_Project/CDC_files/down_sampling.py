import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Load your WNV dataset into a pandas DataFrame
# Load data
df_origin = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/cdc_sum_organism_all_with_phylodiversity.csv",
    index_col=0)

# drop the rows with missing values
df_origin = df_origin.dropna()

# reset the index
df_origin = df_origin.reset_index(drop=True)

# Set the random seed for reproducibility
np.random.seed(3141596)

# Define the desired subsample size
subsample_size = 200

# Perform subsampling based on the state
# for each state, sample n = subsample_size and make sure not all the data with column "Total_Organism_WNV_Count"
# are 0
df_origin = df_origin.groupby("State").apply(
    lambda x: x.sample(n=subsample_size, replace=True)
    if len(x) >= subsample_size
    else x.sample(n=subsample_size, replace=True)
    if x["Total_Organism_WNV_Count"].sum() == 0 else x)

# Reset the index of the subsample DataFrame
df_origin = df_origin.reset_index(drop=True)

# Drop columns that are not needed and assign to df
df = df_origin.drop(columns=["Date", "FIPS", "County_Seat", "Longitude", "Latitude",
                             "Binary_Target", 'Year', 'Month'])

# remove all the comma in the data
df["Population"] = df["Population"].str.replace(",", "")
# convert population to float
df["Population"] = df["Population"].astype(float)


# preprocess data without County, Total_Organism_WNV_Count, State, Year, Month, after standardize data, add back County,Total_Organism_WNV_Count, State, Year, Month
df_preprocessed = df.drop(columns=['County', 'Total_Organism_WNV_Count', 'State'])

# Standardize data
scaler = StandardScaler()
scaler.fit(df_preprocessed)
df_preprocessed = scaler.transform(df_preprocessed)

# add back County, Total_Organism_WNV_Count, State, Year, Month
df_preprocessed_new = pd.DataFrame(df_preprocessed,
                                   columns=df.drop(columns=['County', 'Total_Organism_WNV_Count', 'State']).columns)
df_preprocessed_new['County'] = df['County']
df_preprocessed_new['Total_Organism_WNV_Count'] = df['Total_Organism_WNV_Count']
df_preprocessed_new['State'] = df['State']

# # Build mixed model
fixed_formula = "Total_Organism_WNV_Count ~ State + County + Population + u10_1m_shift + v10_1m_shift + t2m_1m_shift + lai_hv_1m_shift " \
                "+ lai_lv_1m_shift + src_1m_shift + sf_1m_shift + sro_1m_shift + tp_1m_shift + PD"

# vc_formula = {"State": "0 + C(State):C(County)"}

# Fit the mixed effects model
mixed_nested_model = smf.mixedlm(fixed_formula, groups="State", data=df_preprocessed_new)

mixed_nested_model_results = mixed_nested_model.fit()
# Summary of the model
print(mixed_nested_model_results.summary())

# calculate the AIC
print("AIC:", mixed_nested_model_results.aic)

# plot residulas vs fitted values
fig, ax = plt.subplots()
plt.scatter(mixed_nested_model_results.fittedvalues, mixed_nested_model_results.resid)
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.show()
#
# # for each random-effect group, plot residuals vs fitted values
# for name, group in mixed_nested_model_results.resid.groupby(mixed_nested_model_results.groups):
#     fig, ax = plt.subplots()
#     plt.scatter(mixed_nested_model_results.fittedvalues.loc[group.index], group)
#     plt.xlabel("Fitted Values")
#     plt.ylabel("Residuals")
#     plt.title("Residuals vs Fitted Values")
#     # save the plot
#     plt.savefig("/Users/ericliao/Desktop/residuals_vs_fitted_values_" + name + ".png")
#

# normal Q-Q plot of residuals
fig, ax = plt.subplots()
sm.qqplot(mixed_nested_model_results.resid, line='45', ax=ax)
plt.title("Normal Q-Q plot of residuals")
# x and y axis labels
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.show()





