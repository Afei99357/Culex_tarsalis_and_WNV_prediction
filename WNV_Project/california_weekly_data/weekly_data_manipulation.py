import pandas as pd
from scipy.stats import kendalltau

## read in the data
df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/disease_data_weekly_CA/final_file/California_wnv_count_weekly_2011_2012.csv",
    index_col=False)

## separate the date column to year, month, and day columns
df["year"] = df["collect_date"].str.split("/").str[2]
df["month"] = df["collect_date"].str.split("/").str[0]
df["day"] = df["collect_date"].str.split("/").str[1]

## get the bird data
df_bird = df[(df["bird_infection"] == "recent") | (df["bird_infection"] == "Recent")]
df_human = df[df["human_Type"].isna() == False]
df_mosquito = df[df["total_number_in_pool_mos"].isna() == False]

df_human = df_human[df_human["human_Type"] == "WNND"]

## calulate the number of birds by group by county, year and month and keep the collect date column
df_bird_count = df_bird.groupby(["county", "year", "month"]).size().reset_index(name="bird_count")
df_human_count = df_human.groupby(["county", "year", "month"]).size().reset_index(name="human_count")
df_mosquito_count = df_mosquito.groupby(["county", "year", "month"]).size().reset_index(name="mosquito_count")

## output the human count
df_human_count.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
                      "add_0_for_no_wnv/cali_week_only_human_count.csv")

## merge the bird and human data based on county, year, and month
df_merge = pd.merge(df_bird_count, df_human_count, how="outer", on=["county", "year", "month"])

## merge the mosquito data
# df_merge = pd.merge(df_merge, df_mosquito_count, how="outer", on=["county", "year", "month"])

## fill the NaN values with 0 in "human_count" columns
df_merge["human_count"] = df_merge["human_count"].fillna(0)

## add the collect date column

# ## create a column to store the average bird count of the county for each year
# df_merge["bird_count_mean"] = df_merge.groupby(["year", "county"])["bird_count"].transform("mean")
#
# ## fill the NaN values with mean bird count for the bird_count column
# df_merge["bird_count"] = df_merge["bird_count"].fillna(df_merge["bird_count_mean"])


## just fill 0 for the bird count
# df_merge["bird_count"] = df_merge["bird_count"].fillna(0)

## drop na values for birds
# df_merge = df_merge.dropna()

## for each county

kendall_corr, kendall_pvalues = kendalltau(df_merge['bird_count'], df_merge['human_count'])

## print out the correlation and p-value with comments
print("The correlation between bird count and human count is: {}".format(kendall_corr))
print("The p-value is: {}".format(kendall_pvalues))

df_merge.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/Data/California_wnv_count_weekly_2011_2012_origin.csv", index=False)
