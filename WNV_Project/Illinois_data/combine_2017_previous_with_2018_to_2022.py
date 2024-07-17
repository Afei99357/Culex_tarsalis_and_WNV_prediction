import pandas as pd

# import data from 2017 previous
df_2017_previous = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/aggregate_by_county/"
                               "mosquitoes_illinois_aggregate_county_all.csv", index_col=False)

# import data from 2018 to 2022
df_2018_2022 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/county_level_data_2018_2022/"
                           "climate/mos_illinois_2018_2022_climate.csv", index_col=False)

# merge the data
df = pd.concat([df_2017_previous, df_2018_2022], axis=0)

df.to_csv('/Users/ericliao/Desktop/WNV_project_files/illinois_data/aggregate_by_county/all_years/'
          'mos_illinois_county_02_to_22.csv', index=False)