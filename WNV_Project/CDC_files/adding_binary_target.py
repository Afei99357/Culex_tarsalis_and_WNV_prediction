import pandas as pd

# input data for all non-human data
df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/combine_cdc_all_birds.csv",
                 index_col=0)

 # # drop the rows if all four disease column are missing
df = df.dropna(subset=['WNV_Corvid_Count', "WNV_NonCorvid_Count", "Total_Bird_WNV_Count",
                        "Mos_WNV_Count", "Horse_WNV_Count"], how='all')

# fill the missing value with 0
df = df.fillna(0)

# convert County, State to lower case and strip the white space
df['County'] = df['County'].str.lower().str.strip()
df['State'] = df['State'].str.lower().str.strip()

# add one column to add all the disease count together
df['Total_Organism_WNV_Count'] = df['WNV_Corvid_Count'] + df['WNV_NonCorvid_Count'] + df['Total_Bird_WNV_Count'] + \
                                 df['Mos_WNV_Count'] + df['Horse_WNV_Count']

# only get Year, Month, State, FIPS and Total_Organism_WNV_Count
df = df[['Year', 'Month', 'State', 'FIPS', 'Total_Organism_WNV_Count']]

# input data for all county information
df_county = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/"
                        "cdc_sum_organism_fill_centroid.csv", index_col=False)

df_county['Total_Organism_WNV_Count'] = ""

total_orhanism_wmv_count_list = []

# for each record in df_county, if the FIPS, Year, Month exist in df, then add the Total_Organism_WNV_Count
for index, row in df_county.iterrows():
    if len(df.loc[(df['FIPS'] == row['FIPS']) & (df['Year'] == row['Year']) & (df['Month'] == row['Month'])]) > 0:
        # df_county.loc[index, 'Total_Organism_WNV_Count'] = df.loc[(df['FIPS'] == row['FIPS']) &
        #                                                           (df['Year'] == row['Year']) &
        #                                                           (df['Month'] == row['Month']),
        #                                                           'Total_Organism_WNV_Count'].iloc[0]
        total_orhanism_wmv_count_list.append(df.loc[(df['FIPS'] == row['FIPS']) &
                                                                  (df['Year'] == row['Year']) &
                                                                  (df['Month'] == row['Month']),
                                                                  'Total_Organism_WNV_Count'].iloc[0])
        print(index)
    else:
        total_orhanism_wmv_count_list.append(0)
        print(index)
        # df_county.loc[index, 'Total_Organism_WNV_Count'] = 0

df_county['Total_Organism_WNV_Count'] = total_orhanism_wmv_count_list

# add a binary_target column, if the Total_Organism_WNV_Count > 0, then 1, else 0
df_county['binary_target'] = df_county['Total_Organism_WNV_Count'].apply(lambda x: 1 if x > 0 else 0)

# save the data
df_county.to_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/cdc_sum_organism_binary.csv")
