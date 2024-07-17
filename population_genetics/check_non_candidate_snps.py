import pandas as pd

# Load the dataset into a Pandas DataFrame
df_rda_non_can = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/"
                             "Landscape_genetics_GEA/RDA_Redundancy_Analysis/rda_matched_non_candidate_to_gene.csv",
                             index_col=0)

df_lfmm_non_can = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/"
                              "Paper_results/Landscape_genetics_GEA/"
                              "LFMM_LatentFactorMixedModels/lfmm_pc1_matched_non_candidate_to_gene.csv", index_col=0)

df_bayescan_non_can = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/"
                                  "outlier_analysis/bayescan/bayescan_matched_non_candidate_to_gene.csv", index_col=0)

df_pcadapt_non_can = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/"
                                 "Paper_results/outlier_analysis/PCAdapt_RESULTS/"
                                 "pcadapt_matched_non_candidate_to_gene.csv", index_col=0)

## for each dataframe, keep where tyoe is gene
df_rda_non_can = df_rda_non_can[df_rda_non_can["type"] == "gene"]
df_lfmm_non_can = df_lfmm_non_can[df_lfmm_non_can["type"] == "gene"]
df_bayescan_non_can = df_bayescan_non_can[df_bayescan_non_can["type"] == "gene"]
df_pcadapt_non_can = df_pcadapt_non_can[df_pcadapt_non_can["type"] == "gene"]

## get the unqiue number of pairs of seq_nae and location
rda_unique = len(df_rda_non_can[["seq_name", "location"]].drop_duplicates())
lfmm_unique = len(df_lfmm_non_can[["seq_name", "location"]].drop_duplicates())
bayescan_unique = len(df_bayescan_non_can[["seq_name", "location"]].drop_duplicates())
pcadapt_unique = len(df_pcadapt_non_can[["seq_name", "location"]].drop_duplicates())

print("The number of unique pairs of seq_name and location for RDA is: ", rda_unique)
print("The number of unique pairs of seq_name and location for LFMM is: ", lfmm_unique)
print("The number of unique pairs of seq_name and location for Bayescan is: ", bayescan_unique)
print("The number of unique pairs of seq_name and location for PCAdapt is: ", pcadapt_unique)


