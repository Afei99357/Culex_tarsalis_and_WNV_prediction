import statsmodels.api as sm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
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

# replace all the comma in the population
df_origin["Population"] = df_origin["Population"].str.replace(",", "")
# convert population to float
df_origin["Population"] = df_origin["Population"].astype(float)

# Drop columns that are not needed and assign to df
df = df_origin.drop(columns=["Date", "FIPS", "County_Seat", "Longitude", "Latitude",
                             "Binary_Target", 'Year', 'Month'])

# preprocess data without Total_Organism_WNV_Count, after standardize data, add back Total_Organism_WNV_Count
df_preprocessed = df.drop(columns=['Total_Organism_WNV_Count', "State", "County"])

# Standardize data
scaler = StandardScaler()
scaler.fit(df_preprocessed)
df_preprocessed = scaler.transform(df_preprocessed)

# add back Total_Organism_WNV_Count, State and County
df_preprocessed_new = pd.DataFrame(df_preprocessed,
                                   columns=df.drop(columns=['Total_Organism_WNV_Count', "State", "County"]).columns)
df_preprocessed_new['Total_Organism_WNV_Count'] = df['Total_Organism_WNV_Count']
df_preprocessed_new['State'] = df['State']
df_preprocessed_new['County'] = df['County']

# get the unique state
unique_states = df_preprocessed_new['State'].unique()

# for loop through each state and get the data based on state, then fit the data with poisson and negative binomial
# regression model, then calculte the AIC, BIC and likelihood ratio test statistic, store them into a dataframe
for state in unique_states:
    df_state = df_preprocessed_new[df_preprocessed_new['State'] == state]
    df_state = df_state.drop(columns=['State', 'County'])

    # catch the error and warning if fitting the model is failed
    try:
        # Fit Poisson regression model
        poisson_model = sm.GLM(df_state["Total_Organism_WNV_Count"],
                               df_state.drop(columns=['Total_Organism_WNV_Count']),
                               family=sm.families.Poisson()).fit()

        # Fit negative binomial regression model
        negbin_model = sm.GLM(df_state["Total_Organism_WNV_Count"], df_state.drop(columns=['Total_Organism_WNV_Count']),
                              family=sm.families.NegativeBinomial()).fit()
        #catch the error and warning if fitting the model is failed
    except (ValueError, Warning):
        continue

    # Compare the models using AIC
    poisson_aic = poisson_model.aic
    negbin_aic = negbin_model.aic

    # Compare the models using BIC
    poisson_bic = poisson_model.bic
    negbin_bic = negbin_model.bic

    # Calculate log-likelihoods
    poisson_ll = poisson_model.llf
    negbin_ll = negbin_model.llf


    # store the result into a dataframe
    if state == unique_states[0]:
        df_compare = pd.DataFrame(
            {'State': [state], 'Poisson_AIC': [poisson_aic], 'Negative_Binomial_AIC': [negbin_aic],
             'Poisson_BIC': [poisson_bic], 'Negative_Binomial_BIC': [negbin_bic]})
    else:
        df_compare = df_compare.append(
            {'State': state, 'Poisson_AIC': poisson_aic, 'Negative_Binomial_AIC': negbin_aic,
             'Poisson_BIC': poisson_bic, 'Negative_Binomial_BIC': negbin_bic}, ignore_index=True)

# #plot scatter plot for AIC
df_compare.plot.scatter(x='Poisson_AIC', y='Negative_Binomial_AIC')

# # # adding the state name to each point
# for i, txt in enumerate(df_compare['State']):
#     plt.annotate(txt, (df_compare['Poisson_AIC'][i], df_compare['Negative_Binomial_AIC'][i]))

# # add a 45 degree line
x = np.linspace(*plt.xlim())
plt.plot(x, x, color='red')

# # both x and y axis has the same scale
plt.axis('scaled')

# # add title and label
plt.title('AIC Comparison between Poisson and Negative Binomial Regression')

# save the plot
plt.savefig("/Users/ericliao/Desktop/individual_state_comparison_poisson_NB_AIC.png")

# #plot scatter plot for BIC
df_compare.plot.scatter(x='Poisson_BIC', y='Negative_Binomial_BIC')

# # add a 45 degree line
x = np.linspace(*plt.xlim())
plt.plot(x, x, color='red')

# # both x and y axis has the same scale
plt.axis('scaled')

# # add title and label
plt.title('BIC Comparison between Poisson and Negative Binomial Regression')

# save the plot
plt.savefig("/Users/ericliao/Desktop/individual_state_comparison_poisson_NB_BIC.png")

# save the result into a csv file
df_compare.to_csv("/Users/ericliao/Desktop/individual_state_comparison_poisson_NB.csv")