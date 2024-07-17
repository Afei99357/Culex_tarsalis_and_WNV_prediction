import pandas as pd
import os

## read file with candidate snps from lfmm
pca_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/Landscape_genetics/"
    "LFMM_LatentFactorMixedModels/lfmm_pc1_matched_candidate_to_gene.csv",
    header=0,
    index_col=0,
    sep=",",
)

## read file with candidate snps from rda
bayescan_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/"
    "Landscape_genetics/RDA_Redundancy_Analysis/rda_matched_candidate_to_gene.csv",
    header=0,
    index_col=0,
    sep=",",
)


## get the overlapping snps between lfmm and rda based on seq_name, start and end
overlapping_snps_df = pd.merge(
    pca_df,
    bayescan_df,
    how="inner",
    on=["seq_name", "type", "start", "end", "attributes"],
    suffixes=("_lfmm", "_rda"),
)


## only keep columns: seq_name, location_lfmm, location_rda, type, start, end and attributes
overlapping_snps_df = overlapping_snps_df[
    ["seq_name", "location_lfmm", "location_rda", "type", "start", "end", "attributes"]
]

## only keep the type == gene
overlapping_snps_df = overlapping_snps_df[overlapping_snps_df["type"] == "gene"]

overlapping_snps_df.to_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/"
    "Landscape_genetics/Landscape_genetics_results_merge_lfmm_rda/"
    "lfmm_pc1_rda_overlapping_snps_within_gene.csv",
    header=True,
    index=False,
    sep=",",
)

## create a new dataframe to store the sift result
sift_result_df = pd.DataFrame()

# read all the sift tsv files names in the directory
tsv_file_list = []
dir = "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/sift_prediciton_results/sift_prediction_results"
for file in os.listdir(dir):
    if file.endswith(".tsv"):
        tsv_file_list.append(file)

row_list = []
# for each row in overlapping_snps_df, find the tsv file which file name contains the seq_name,
# in the tsv file, find the second column value of the tsv file is the same as location_lfmm of the row, save the row of tsv and append a new column to the row, value is lfmm
# or the second column value of the tsv file  is the same as location_rda of the row, save the row of tsv and append a new column to the row, value is rda
for index, row in overlapping_snps_df.iterrows():
    seq_name = row["seq_name"]
    location_lfmm = row["location_lfmm"]
    location_rda = row["location_rda"]
    for tsv_file in tsv_file_list:
        if seq_name in tsv_file:
            tsv_file_df = pd.read_csv(
                os.path.join(dir, tsv_file), header=None, index_col=False, sep="\t"
            )
            for tsv_index, tsv_row in tsv_file_df.iterrows():
                if tsv_row[1] == location_lfmm:
                    tsv_row[17] = "lfmm"
                    row_list.append(tsv_row)
                elif tsv_row[1] == location_rda:
                    tsv_row[17] = "rda"
                    row_list.append(tsv_row)

## append row_list to sift_result_df
sift_result_df = sift_result_df.append(row_list)

## rename the columns of sift_result_df
sift_result_df.columns = [
    "CHROM",
    "POSITION",
    "REF_ALLELE",
    "ALT_ALLELE",
    "TRANSCRIPT_ID",
    "GENE_ID",
    "GENE_NAME",
    "REGION",
    "VARIANT_TYPE",
    "REF_AA",
    "ALT_AA",
    "AA_POS",
    "SIFT_SCORE",
    "SIFT_MEDIAN",
    "NUM_SEQs",
    "dbSNP",
    "PREDICTION",
    "candidate_source",
]

## remove the duplicates in sift_result_df
sift_result_df = sift_result_df.drop_duplicates()

## rearrange the columns of sift_result_df to put the candidate_source column at the first column
sift_result_df = sift_result_df[
    [
        "candidate_source",
        "CHROM",
        "POSITION",
        "REF_ALLELE",
        "ALT_ALLELE",
        "TRANSCRIPT_ID",
        "GENE_ID",
        "GENE_NAME",
        "REGION",
        "VARIANT_TYPE",
        "REF_AA",
        "ALT_AA",
        "AA_POS",
        "SIFT_SCORE",
        "SIFT_MEDIAN",
        "NUM_SEQs",
        "dbSNP",
        "PREDICTION",
    ]
]

sift_result_df.to_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/"
    "sift_prediciton_results/match_snps_sift_results/lfmm_pc1_rda_overlapping_snps_sift_results.csv",
    header=True,
    index=False,
    sep=",",
)
