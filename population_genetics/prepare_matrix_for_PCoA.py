import pandas as pd

# read tab delimited file
df_interproscan = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/pairwise_fst_table.csv",
    header=0,
    index_col=False,
    sep=",",
)

# get the unique population list from both Pop1 and Pop2 columns
unique_population_list = list(
    set(df_interproscan["Pop1"].tolist() + df_interproscan["Pop2"].tolist())
)

# generate a matrix with the same number of rows and columns as the number of populations from unique population list
df_matrix = pd.DataFrame(index=unique_population_list, columns=unique_population_list)

# fill the diagonal with 0 and the rest with NA
for i in range(len(unique_population_list)):
    for j in range(len(unique_population_list)):
        if i == j:
            df_matrix.iloc[i, j] = 0
        else:
            df_matrix.iloc[i, j] = "NA"

# fill the matrix with the pairwise fst values
for index, row in df_interproscan.iterrows():
    df_matrix.loc[row["Pop1"], row["Pop2"]] = row["PairwiseFst"]
    df_matrix.loc[row["Pop2"], row["Pop1"]] = row["PairwiseFst"]

# save the df_matrix as a csv file
df_matrix.to_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/pairwise_fst_matrix_pcoa.csv"
)
