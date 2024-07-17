import pandas as pd

# load data
df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/cdc_sum_organism_all_binary.csv",
                 index_col=0)

df_phylodiversity = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/birds_data/"
                                "estimates_of_avian_phylodiversity/avi_phylodiv_wnv_041822.csv", index_col=0)

# based on the FIPS in df and STCO_FIPS in df_phylodiversity, merge the phylodiversity index to the cdc dataset
df = df.merge(df_phylodiversity, left_on="FIPS", right_on="STCO_FIPS", how="left")

# drop the STCO_FIPS column
df = df.drop(columns="STCO_FIPS")

# save the data
df.to_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/"
          "cdc_sum_organism_all_with_phylodiversity.csv")