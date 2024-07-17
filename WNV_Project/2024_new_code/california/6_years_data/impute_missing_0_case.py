import pandas as pd

## import ca data
df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/data/cali_week_wnnd.csv", index_col=False, header=0)

## replace 0 values with nan for "Human_WNND_Count", "WNV_Mos_count", "WNV_Corvid_Count", "WNV_NonCorvid_Count" and "TotalBird_WNV_Count"
df["Human_WNND_Count"] = df["Human_WNND_Count"].replace(0, float("nan"))
df["WNV_Mos_Count"] = df["WNV_Mos_Count"].replace(0, float("nan"))
df["WNV_Corvid_Count"] = df["WNV_Corvid_Count"].replace(0, float("nan"))
df["WNV_NonCorvid_Count"] = df["WNV_NonCorvid_Count"].replace(0, float("nan"))
df["Total_Bird_WNV_Count"] = df["Total_Bird_WNV_Count"].replace(0, float("nan"))

## for each row in the df, if "Human_WNND_Count", "WNV_Mos_count", "WNV_Corvid_Count", "WNV_NonCorvid_Count" and "TotalBird_WNV_Count" are nan, fill 0 with "Human_WNND_Count"
df.loc[df["Human_WNND_Count"].isna() & df["WNV_Mos_Count"].isna() & df["WNV_Corvid_Count"].isna() & df["WNV_NonCorvid_Count"].isna() & df["Total_Bird_WNV_Count"].isna(), "Human_WNND_Count"] = 0

## adding an column for "average_human_case_monthly" by grouping the data by "FIPS" and 'Year' and calculate the 12 month mean of "Reported_human_cases"
df["average_human_case_monthly"] = df.groupby(["FIPS", "Year"])["Human_WNND_Count"].transform("sum") / 12

## for each row, if the "Human_WNND_Count" is nan, fill "Human_WNND_Count" with "average_human_case_monthly"
df.loc[df["Human_WNND_Count"].isna(), "Human_WNND_Count"] = df["average_human_case_monthly"]

## output the dataframe to a csv file
df.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/data/cali_week_wnnd_impute_monthly_mean_value.csv", index=False)

print(df.columns.values)





