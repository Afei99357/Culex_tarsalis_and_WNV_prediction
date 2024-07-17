import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM
import statsmodels.discrete.count_model as cm
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load data
df_origin = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/cdc_sum_organism_all_with_phylodiversity.csv",
    index_col=0)

# Drop columns that are not needed and assign to df
df = df_origin.drop(columns=["Date", "FIPS", "County_Seat", "Longitude", "Latitude",
                             "Binary_Target", 'Year', 'Month', 'State'])

# remove all the comma in the data
df["Population"] = df["Population"].str.replace(",", "")
# convert population to float
df["Population"] = df["Population"].astype(float)

# Drop rows with missing values and reset index
df = df.dropna()
df = df.reset_index(drop=True)

# find out the percentage of non-0 target value
print("Percentage of non-0 target value: ", len(df[df["Total_Organism_WNV_Count"] != 0]) / len(df))

# preprocess data without County, Total_Organism_WNV_Count, State, Year, Month, after standardize data, add back County,Total_Organism_WNV_Count, State, Year, Month
df_preprocessed = df.drop(columns=['County', 'Total_Organism_WNV_Count'])

# Standardize data
scaler = StandardScaler()
scaler.fit(df_preprocessed)
df_preprocessed = scaler.transform(df_preprocessed)

# add back County, Total_Organism_WNV_Count, State, Year, Month
df_preprocessed_new = pd.DataFrame(df_preprocessed,
                                   columns=df.drop(columns=['County', 'Total_Organism_WNV_Count']).columns)
df_preprocessed_new['County'] = df['County']
df_preprocessed_new['Total_Organism_WNV_Count'] = df['Total_Organism_WNV_Count']

# create the formula for the GLMM model where random effect is the state and county
fml = "Total_Organism_WNV_Count ~ Population + u10_1m_shift + v10_1m_shift + t2m_1m_shift + lai_hv_1m_shift " \
                "+ lai_lv_1m_shift + src_1m_shift + sf_1m_shift + sro_1m_shift + tp_1m_shift + PD"

vc_fml = {"z1": "0 + C(County)"}
glmm2 = PoissonBayesMixedGLM.from_formula(fml, vc_fml, df_preprocessed_new)

# fit the model
res2 = glmm2.fit_vb()

# print the summary
print(res2.summary())
