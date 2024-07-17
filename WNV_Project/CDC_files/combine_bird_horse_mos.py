import pandas as pd

# # load bird
data_bird = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/"
                        "Dead_Bird_Denominator_Numerator_monthly.csv", index_col=0)

# # load mosquitoes
data_mos = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/"
                       "Mosquito_Denominator_Numerator_WNV_monthly.csv", index_col=0)

# # load horse
data_horse_numerator = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/"
                                   "Horse_Numerator_monthly.csv", index_col=0)

# # based on Year, State, Month, and County, merge the three datasets
data_merge = pd.merge(data_bird, data_mos, how='outer', on=['Year', 'State', 'Month', 'County'],
                      suffixes=('_bird', '_mos'))

# drop CorvidsReported, CorvidsTested, OtherReported, OtherTested, Collected and Tested columns
data_merge = data_merge.drop(["CorvidsReported", "CorvidsTested", "OtherReported", "OtherTested", "Collected", "Tested"], axis=1)

# # merge horse
data_merge = pd.merge(data_merge, data_horse_numerator, how='outer', on=['Year', 'State', 'Month', 'County'],
                        suffixes=('', '_horse'))

## load national disease data to fill in the other information
data_disease = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/"
                           "human_neuroinvasive_wnv_rate_log_population_correct_poverty_mean_green_index.csv", index_col=False)

# # create a column list for choose from
col_list = ["FIPS", "County", "State", "State_Code", "Year", "County_Seat_Latitude",
            "County_Seat_Longitude", "Population", "Poverty_Estimate_All_Ages", "Land_Area_2010"]

# # choose the columns from the national disease data
data_disease = data_disease[col_list]

# # get population and poverty information by year
data_pop_poverty = data_disease[['FIPS', 'State_Code', 'Year', 'Population', 'Poverty_Estimate_All_Ages']]

# # drop the duplicates
data_pop_poverty = data_pop_poverty.drop_duplicates()

# # merge the population and poverty information to the data_merge
data_merge = pd.merge(data_merge, data_pop_poverty, how='left', left_on=['County', 'State',  'Year'],
                      right_on=['FIPS', 'State_Code', 'Year'])

# # # get the rows where the value of FIPS and State_Code column is NaN
# data_merge_missing = data_merge[data_merge['FIPS'].isnull() & data_merge['State_Code'].isnull()]
#
# # # get the wrong FIPS and State_Code
# data_modify_needed = data_merge_missing[['County', 'State']]
#
# data_modify_unique = data_modify_needed.drop_duplicates()
#
# data_modify_unique.to_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/wrong_fips.csv", index=False)

# # drop State, County from data_merge
data_merge = data_merge.drop(["State_Code", "FIPS"], axis=1)

# # drop Population and Poverty_Estimate_All_Ages columns
data_disease = data_disease.drop(["Population", "Poverty_Estimate_All_Ages"], axis=1)

# # drop duplicates
data_disease = data_disease.drop_duplicates()

# # merge the data_merge and data_disease
data_merge_final = pd.merge(data_merge, data_disease, how='left', left_on=['Year', 'State', 'County'],
                            right_on=['Year', "State_Code", "FIPS"], suffixes=('', '_national'))

# # # get the rows where the value of FIPS and State_Code column is NaN
# data_merge_missing = data_merge_final[data_merge_final['FIPS'].isnull() & data_merge_final['State_Code'].isnull()]
#
# # # remove State is AK and HI
# data_merge_missing = data_merge_missing[data_merge_missing['State'] != 'AK']
# data_merge_missing = data_merge_missing[data_merge_missing['State'] != 'HI']
#
# # # get unique values
# data_merge_missing_final = data_merge_missing.drop_duplicates()
#
# # #
# # data_merge_missing_final.to_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/wrong-fips.csv", index=False)

# # LOAD THE DATA WITH CORRECT FIPS
data_correct_fips = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/wrong-fips_population.csv", index_col=False)

# # # load population data
# data_population = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/demographic/population_with_fips_year.csv", index_col=False)
#
# # merge data-correct_fips and data_population based on year and FIPS
# data_fips_new = pd.merge(data_correct_fips,data_population, how="left", left_on=['Year', 'County'], right_on=['Year', 'FIPS'])
#
# data_fips_new.to_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/wrong-fips_population.csv", index=False)


# # for each row in data_correct_fips, find the corresponding row in data_merge_final, and fill in the missing values for County_Seat_Longitude, Counry_Seat_Latitude and population columns
for index, row in data_correct_fips.iterrows():
    data_merge_final.loc[(data_merge_final['Year'] == row['Year']) & (data_merge_final['State'] == row['State']) & (
            data_merge_final['County'] == row['County']), 'County_Seat_Longitude'] = row['County_Seat_Longitude']
    data_merge_final.loc[(data_merge_final['Year'] == row['Year']) & (data_merge_final['State'] == row['State']) & (
            data_merge_final['County'] == row['County']), 'County_Seat_Latitude'] = row['County_Seat_Latitude']
    data_merge_final.loc[(data_merge_final['Year'] == row['Year']) & (data_merge_final['State'] == row['State']) & (
                data_merge_final['County'] == row['County']), 'Population'] = row['Population']

# # for each values in Population column, remove the comma
for index, row in data_merge_final.iterrows():
    if type(row['Population']) == str:
        data_merge_final.loc[index, 'Population'] = row['Population'].replace(',', '')

# # convert the Population column to float
data_merge_final['Population'] = data_merge_final['Population'].astype(float)

data_merge_final['Population']

