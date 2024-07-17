import pandas as pd

common_sift_df_deleterious = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/"
    "sift_prediciton_results/match_snps_sift_results/"
    "overlapping_landscape_outlier_analysis_snp_sift_prediction_deleterious.csv",
    header=0,
    index_col=0,
    sep=",",
)

# ## get gene list and build gene_list value by append 20471- in front of each gene name and .m01 at the end
gene_list = []
for gene_entry in common_sift_df_deleterious["TRANSCRIPT_ID"]:
    gene_list.append("20471-" + gene_entry)

## look up the gene name from the gene list
df_interproscan = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/original_vcf/"
    "Culex-tarsalis-v1.0.a1.5d6405151b078-interproscan.tab",
    header=8,
    index_col=False,
    sep="\t",
)

# from df_interproscan, the Name column contains the characters that we want to match in the gene_list. get the rows that contains gene_list values as part of the valu
df_interproscan_gene = df_interproscan[df_interproscan["Name"].isin(gene_list)]

df_interproscan_gene.to_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/"
    "compare_landscape_and_outlier_analysis/"
    "landscape_outlier_analysis_common_gene_sift_deleterious.csv",
    index=False,
)
