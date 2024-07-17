import pandas as pd
import os

# read the population culex gene ontology info
culex_gene_ontology_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/"
    "compare_landscape_and_outlier_analysis/overlap_unique_gene_ontology_info.csv",
    header=0
)

## read rda result
rda_result_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/Landscape_genetics_GEA/"
    "RDA_Redundancy_Analysis/rda_matched_candidate_to_gene.csv",
    header=0,
    index_col=0,
)

## subset the rda_result_df to only include the type is gene
rda_result_df = rda_result_df[rda_result_df["type"] == "gene"]

## based on the Gene_ID column in culex_gene_ontology_df, find the rows in rda_result_df which column attributes contains the Gene_ID
rda_result_df = rda_result_df[
    rda_result_df["attributes"].str.contains("|".join(culex_gene_ontology_df["Gene_ID"]))
]

## subset the rda_result_df to only include the seq_name and location columns
rda_result_df = rda_result_df[["seq_name", "location", "attributes"]]

## based on seq_name and location columns, drop the duplicated rows
rda_result_df = rda_result_df.drop_duplicates(subset=["seq_name", "location"])

## add a new column called alternate_allele
rda_result_df["alternate_allele"] = ""

## get the unique seq_name list from the seq_name column in rda_result_df
seq_name_list = list(set(rda_result_df["seq_name"].tolist()))

# read all the sift tsv files names in the directory
tsv_file_list = []
for file in os.listdir(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/sift_prediciton_results/sift_prediction_results"
):
    if file.endswith(".tsv"):
        ## if the file name contains any of the seq_name in the seq_name_list, then add the file name to the tsv_file_list
        for seq_name in seq_name_list:
            if seq_name in file:
                tsv_file_list.append(file)

## read tsv_file, then find rows that second column value is the same as location, the fourth column is the alternate_allele,
for tsv_file in tsv_file_list:
    sift_df = pd.read_csv(
        "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/sift_prediciton_results/"
        "sift_prediction_results/" + tsv_file,
        header=None,
        sep="\t",
        index_col=None,
    )
    ## get the seq_name from the tsv_file
    seq_name = tsv_file.split("SIFTPredictions_")[1].split("_")[0]
    ## based on the seq_name, get the rows from rda_result_df
    rda_result_df_sub = rda_result_df[rda_result_df["seq_name"] == seq_name]

    ## for each location in the rda_result_df_sub, find the rows that second column value is the same as location,
    ## then save the fourth column value to the alternate_allele column in rda_result_df
    for index, row in rda_result_df_sub.iterrows():
        rda_result_df.at[index, "alternate_allele"] = sift_df[sift_df[1] == row["location"]][3].tolist()[0]
        ## # constrcut the alternate_allele value as seq_name_location.alternate_allele,
        # save the value to the new column
        rda_result_df.at[index, "alternate_allele"] = row["seq_name"] + "_" + str(row["location"]) + "." + rda_result_df.at[index, "alternate_allele"]

## save the rda_result_df as a csv file and reindex the rows
rda_result_df.to_csv(
    "/Users/ericliao/Desktop/WNV_project_files/WNV/california/data/culex_pop_gene_snp_candidates.csv",
    index=False
)

