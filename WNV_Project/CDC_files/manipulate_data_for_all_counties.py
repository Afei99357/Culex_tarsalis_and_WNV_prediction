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
df['Total_Organism_WNV_Count'] = df['WNV_Corvid_Count'] + df['WNV_NonCorvid_Count'] + df['Mos_WNV_Count'] + \
                                 df['Horse_WNV_Count']

# in df, if the FIPS is the same, but County are different, then change the County to the same as the first one
for fips in df['FIPS'].unique():
    df.loc[df['FIPS'] == fips, 'County'] = df.loc[df['FIPS'] == fips, 'County'].iloc[0]

# if in df, same FIPS has different State, then change the State to the one that is not nan
for fips in df['FIPS'].unique():
    if len(df.loc[df['FIPS'] == fips, 'State'].unique()) > 1:
        df.loc[df['FIPS'] == fips, 'State'] = df.loc[df['FIPS'] == fips, 'State'].dropna().iloc[0]

# # get unique pairs of FIPS, County and State
unique_fips = set(df[['FIPS', 'County', 'State']].apply(tuple, axis=1))

# based on unique data, create an empty dataframe include each county, every year from 2003 to 2021, for each month
df_empty = pd.DataFrame(columns=['Year', 'Month', 'FIPS', 'County', 'State'])

df_list = []

# for each unique FIPS, create row with year from 2003 to 2021, and month from January to December
row_index = 0
for fips in unique_fips:
    for year in range(2003, 2022):
        for month in range(1, 13):
            df_list.append({'Year': year, 'Month': month, 'FIPS': fips[0], 'County': fips[1], 'State': fips[2]})
            print(row_index)
            row_index = row_index + 1

# finish creating the new table
df_empty = df_empty.append(df_list)

# add more feature columns
# df_empty['Poverty_Estimate_All_Ages'] = ''

# read population data file
df_population = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/demographic/population_with_fips_year.csv",
                            index_col=False)

# Merge df_empty with df_population based on FIPS and Year
df_empty = pd.merge(df_empty, df_population, how='left', left_on=['FIPS', 'Year'], right_on=['FIPS', 'Year'])

# # load county seat coordinate data
df_county_seat = pd.read_csv("/Users/ericliao/Desktop/County_County_seat_FIPS_info.csv", index_col=False)

# # remove the County at the end of the County column
df_county_seat['County'] = df_county_seat['County'].str.split(' County').str[0]
df_county_seat['County'] = df_county_seat['County'].str.lower().str.strip()

# # drop duplicate rows where the State_Code and County are the same
df_county_seat = df_county_seat.drop_duplicates(subset=['State', 'County', 'FIPS', 'County_Seat'])

# convert both County to lower case and strip the white space
df_empty['County'] = df_empty['County'].str.lower().str.strip()
df_county_seat['County'] = df_county_seat['County'].str.lower().str.strip()
df_county_seat['State'] = df_county_seat['State'].str.lower().str.strip()


# # merge df_empty with df_county_seat based on County and State
df_empty = pd.merge(df_empty,
                    df_county_seat,
                    how='left',
                    on=['County', 'State', 'FIPS'])

# # output the df_empty to csv file
df_empty.to_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/cdc_sum_organism.csv")

# # load the cdc_sum_organism.csv file

