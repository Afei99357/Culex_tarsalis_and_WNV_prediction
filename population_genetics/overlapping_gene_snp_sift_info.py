import pandas as pd

# read file with common gene from all the methods
df_common_gene = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/"
                             "compare_landscape_and_outlier_analysis/overlap_unique_gene_ontology_info.csv",
                             header=0, sep=",", index_col=False)

## get the list of common genes
common_gene_list = df_common_gene["Gene_ID"].tolist()

# read snp sift prediction from pca
pca_sift_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/sift_prediciton_results/"
    "pcadapt_candidates_sift_prediction.csv",
    header=0,
    index_col=False,
    sep=",",
)

# read snp sift prediction from bayescan
bayescan_sift_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/sift_prediciton_results/"
    "bayescan_candidates_sift_prediction.csv",
    header=0,
    index_col=False,
    sep=",",
)

# read snp sift prediction from lfmm pc1
lfmm_pc1_sift_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/sift_prediciton_results/"
    "lfmm_pc1_candidates_sift_prediction.csv",
    header=0,
    index_col=False,
    sep=",",
)

# read snp sift prediction from rda
rda_sift_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/sift_prediciton_results/"
    "rda_candidates_sift_prediction.csv",
    header=0,
    index_col=False,
    sep=",",
)

## for each df, find the rows that GENE_ID is in the common_gene_list
pca_sift_df = pca_sift_df[pca_sift_df["GENE_ID"].isin(common_gene_list)]
bayescan_sift_df = bayescan_sift_df[bayescan_sift_df["GENE_ID"].isin(common_gene_list)]
lfmm_pc1_sift_df = lfmm_pc1_sift_df[lfmm_pc1_sift_df["GENE_ID"].isin(common_gene_list)]
rda_sift_df = rda_sift_df[rda_sift_df["GENE_ID"].isin(common_gene_list)]

## combine all the sift dfs with extra column to indicate the method
pca_sift_df["method"] = "pcadapt"
bayescan_sift_df["method"] = "bayescan"
lfmm_pc1_sift_df["method"] = "lfmm_pc1"
rda_sift_df["method"] = "rda"

## combine all the sift dfs
combined_sift_df = pd.concat([pca_sift_df, bayescan_sift_df, lfmm_pc1_sift_df, rda_sift_df])

## output the combined sift df to a csv file
combined_sift_df.to_csv("/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/"
                        "overlapping_gene_snps_sift_info_combined.csv", header=True, index=False, sep=",")
