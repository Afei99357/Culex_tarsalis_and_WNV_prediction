import statsmodels.api as sm
from scipy import stats
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load your WNV dataset into a pandas DataFrame
# Load data
df_origin = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/cdc_sum_organism_all_with_phylodiversity.csv",
    index_col=0)

# drop the rows with missing values
df_origin = df_origin.dropna()

# reset the index
df_origin = df_origin.reset_index(drop=True)

# replace all the comma in the population
df_origin["Population"] = df_origin["Population"].str.replace(",", "")
# convert population to float
df_origin["Population"] = df_origin["Population"].astype(float)

# Drop columns that are not needed and assign to df
df = df_origin.drop(columns=["State", "County", "Date", "FIPS", "County_Seat", "Longitude", "Latitude",
                             "Binary_Target", 'Year', 'Month'])

# preprocess data without Total_Organism_WNV_Count, after standardize data, add back Total_Organism_WNV_Count
df_preprocessed = df.drop(columns=['Total_Organism_WNV_Count'])

# Standardize data
scaler = StandardScaler()
scaler.fit(df_preprocessed)
df_preprocessed = scaler.transform(df_preprocessed)

# add back Total_Organism_WNV_Count
df_preprocessed_new = pd.DataFrame(df_preprocessed, columns=df.drop(columns=['Total_Organism_WNV_Count']).columns)
df_preprocessed_new['Total_Organism_WNV_Count'] = df['Total_Organism_WNV_Count']

# Fit Poisson regression model
poisson_model = sm.GLM(df_preprocessed_new["Total_Organism_WNV_Count"], df_preprocessed_new.drop(columns=['Total_Organism_WNV_Count']), family=sm.families.Poisson()).fit()

# Fit negative binomial regression model
negbin_model = sm.GLM(df_preprocessed_new["Total_Organism_WNV_Count"], df_preprocessed_new.drop(columns=['Total_Organism_WNV_Count']), family=sm.families.NegativeBinomial()).fit()


# Compare the models using AIC
poisson_aic = poisson_model.aic
negbin_aic = negbin_model.aic

# Print the AIC values
print("AIC for Poisson model:", poisson_aic)
print("AIC for Negative Binomial model:", negbin_aic)

# Calculate log-likelihoods
poisson_ll = poisson_model.llf
negbin_ll = negbin_model.llf

# Calculate the likelihood ratio test statistic
lr_test_stat = -2 * (poisson_ll - negbin_ll)

# Calculate the p-value using chi-square distribution
p_value = 1 - stats.chi2.cdf(lr_test_stat, df=1)  # df=1 for one degree of freedom

# Print the likelihood ratio test statistic and p-value
print("Likelihood Ratio Test Statistic:", lr_test_stat)
print("P-value:", p_value)
