import pandas as pd

df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/outlier_analysis/PCAdapt_RESULTS/pcadapt_matched_candidate_to_gene.csv",
                 header=0, sep=",", index_col=False)

## get only type is gene
df_gene = df[df["type"] == "gene"]

## get the unique pair of seq_name and location
df_gene_unique = df_gene[["seq_name", "location"]].drop_duplicates()

print(df_gene_unique.shape)