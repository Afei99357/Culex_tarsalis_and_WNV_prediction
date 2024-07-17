import pandas as pd

# read the information of samples
gene_match_info_rda_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/Landscape_genetics_GEA/RDA_Redundancy_Analysis/rda_matched_candidate_to_gene.csv",
    header=0,
    index_col=0,
)

## get only type is gene
gene_match_info_rda_df = gene_match_info_rda_df[gene_match_info_rda_df["type"] == "gene"]

# ## get overlapping genes
# overlapping_genes_df = pd.read_csv(
#     "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/compare_landscape_and_outlier_analysis/"
#     "overlap_unique_gene_ontology_info.csv",
#     header=0,
#     index_col=False,
# )
#
# ## get the gene list from the overlapping_genes_df
# gene_list = overlapping_genes_df["Gene_ID"].tolist()

## choose gene list manually
gene_list = ["Ct.00g030230", "Ct.00g049290", "Ct.00g095350", "Ct.00g154760"]

## find all the rows that in the attributes column of gene_match_info_rda_df, the values contains any of the gene_list
gene_match_info_rda_df = gene_match_info_rda_df[
    gene_match_info_rda_df["attributes"].str.contains("|".join(gene_list))
]

## create a new dataframe to store the values which is built with value in seq_name and location. put a _ to connect them as new value to store
new_df = pd.DataFrame()
new_df["snp"] = [
    seq_name + "_" + str(location)
    for seq_name, location in zip(
        gene_match_info_rda_df["seq_name"], gene_match_info_rda_df["location"]
    )
]

## output the new_df to a csv file
new_df.to_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/compare_landscape_and_outlier_analysis/"
    "candidate_genes_match_to_snps_list.csv"
)


