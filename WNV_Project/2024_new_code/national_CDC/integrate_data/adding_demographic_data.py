import pandas as pd

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/cdc_human_1999_to_2023/"
                   "WNV_human_and_non_human_yearly_climate.csv",
                   index_col=False,
                   header=0)

## load dataset has population, land area, and bird info
df_population = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/monthly/"
                            "cdc_sum_organism_all_with_phylodiversity.csv",
                            index_col=0, header=0)

## only keep the columns that are needed, Year, FIPS, Population, and PD
df_population = df_population[["Year", "FIPS", "Population", "PD"]]

## remove the duplicate rows based on Year and FIPS
df_population = df_population.drop_duplicates(subset=["Year", "FIPS"])

## merge the population and PD data with the original data
data = pd.merge(data, df_population, on=["Year", "FIPS"], how="left")

## save the data to a csv file
data.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/cdc_human_1999_to_2023/"
            "WNV_human_and_non_human_yearly_climate_demographic_bird.csv", index=False)
