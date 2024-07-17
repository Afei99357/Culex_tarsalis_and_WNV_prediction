import pandas as pd
import os
import matplotlib.pyplot as plt

# read the population culex gene ontology info
culex_gene_ontology_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/"
    "compare_landscape_and_outlier_analysis/overlap_unique_gene_ontology_info.csv",
    header=0
)

## read rda result
alt_allel_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/Landscape_genetics_GEA/"
    "RDA_Redundancy_Analysis/rda_matched_candidate_to_gene.csv",
    header=0,
    index_col=0,
)

## subset the alt_allel_df to only include the type is gene
alt_allel_df = alt_allel_df[alt_allel_df["type"] == "gene"]

## based on the Gene_ID column in culex_gene_ontology_df, find the rows in alt_allel_df which column attributes contains the Gene_ID
alt_allel_df = alt_allel_df[
    alt_allel_df["attributes"].str.contains("|".join(culex_gene_ontology_df["Gene_ID"]))
]

## subset the alt_allel_df to only include the seq_name and location columns
alt_allel_df = alt_allel_df[["seq_name", "location", "attributes"]]

## based on seq_name and location columns, drop the duplicated rows
alt_allel_df = alt_allel_df.drop_duplicates(subset=["seq_name", "location"])

## add a new column called alternate_allele
alt_allel_df["alternate_allele"] = ""

## get the unique seq_name list from the seq_name column in alt_allel_df
seq_name_list = list(set(alt_allel_df["seq_name"].tolist()))

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
    ## based on the seq_name, get the rows from alt_allel_df
    alt_allel_df_sub = alt_allel_df[alt_allel_df["seq_name"] == seq_name]

    ## for each location in the alt_allel_df_sub, find the rows that second column value is the same as location,
    ## then save the fourth column value to the alternate_allele column in alt_allel_df
    for index, row in alt_allel_df_sub.iterrows():
        alt_allel_df.at[index, "alternate_allele"] = sift_df[sift_df[1] == row["location"]][3].tolist()[0]
        ## # constrcut the alternate_allele value as seq_name_location.alternate_allele,
        # save the value to the new column
        alt_allel_df.at[index, "alternate_allele"] = row["seq_name"] + "_" + str(row["location"]) + "." + alt_allel_df.at[index, "alternate_allele"]

## in the attributes column, get the gene name by split the string by the first ; and get the value after the = sign
alt_allel_df["gene_name"] = alt_allel_df["attributes"].apply(lambda x: x.split(";")[0].split("=")[1])


## read snp info
allelic_frequency_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/WNV/california/data/allele_frequency_table_sample_candidate_gene_all.csv",
    header=0,
    index_col=0
)

## read population info
population_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/"
                            "Ctarsalis_sample_w_GPS_climate_average_new_filtered_id_region.csv",
                            header=0,
                            index_col=0)

## subset the population_df to only include the vcfID, locID, region, popID, GPS.Lat and GPS.Lon
population_df = population_df[["vcfID", "popID", "GPS.Lat", "GPS.Lon"]]

## merge population_df and allelic_frequency_df based on popID
snp_df = pd.merge(allelic_frequency_df, population_df, how="left", on=["popID"])

## rearrange the columns, put vcfID, locID, region, popID, GPS.Lat, GPS.Lon first and the oher columns after
snp_df = snp_df[["vcfID", "popID", "GPS.Lat", "GPS.Lon"] + [col for col in snp_df.columns if col not in ["vcfID", "locID", "region", "popID", "GPS.Lat", "GPS.Lon"]]]

## rename GPS.Lat to latitude and GPS.Lon to longitude
snp_df = snp_df.rename(
    columns={"GPS.Lat": "Latitude", "GPS.Lon": "Longitude"}
)

## choose the rows are in the interesting gene list
## create a new list to store the interesting gene list
interest_gene_list = ["Ct.00g049290", "Ct.00g030230", "Ct.00g095350", "Ct.00g154760"]

## based on the gene_name column in alt_allel_df, find the rows that the gene_name is in the interest_gene_list
alt_allel_df = alt_allel_df[alt_allel_df["gene_name"].isin(interest_gene_list)]

## output the alt_allel_df to a csv file
alt_allel_df.to_csv("/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/"
                    "compare_landscape_and_outlier_analysis/compare_candidate_allel_alt_allele_freq_over_samples/"
                    "culex_pop_gene_snp_candidates.csv", index=False)

## ## read allel frequency info
allelic_frequency_df = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/WNV/california/data/allele_frequency_table_sample_candidate_gene_all.csv",
    header=0,
    index_col=0
)

## read population info
population_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/"
                            "Ctarsalis_sample_w_GPS_climate_average_new_filtered_id_region.csv",
                            header=0,
                            index_col=0)

## adding a new column in allelic_frequency_df called region
allelic_frequency_df["region"] = ""

## based on the popID in the allelic_frequency_df, get the region value from the population_df
for index, row in allelic_frequency_df.iterrows():
    allelic_frequency_df.at[index, "region"] = population_df[population_df["popID"] == row["popID"]]["region"].tolist()[0]

## sort the allelic_frequency_df by the region column in descending order
allelic_frequency_df = allelic_frequency_df.sort_values(by=["region"], ascending=False)

region_to_color = {
    "Northwest": "skyblue",
    "Midwest": "hotpink",
    "West Coast": "goldenrod",
    "Southwest": "forestgreen",
}

### for each value in the column alternate_allele of alt_allel_df, find the column name in the allelic_frequency_df.
## Then plot bar plot for each column, with x axis is the popID name, y axis is the frequency and save the histogram to a pdf file
for index, row in alt_allel_df.iterrows():
    ## get the alternate_allele value
    alternate_allele = row["alternate_allele"]
    ## based on the alternate_allele value, get the column name in the allelic_frequency_df
    column_name = allelic_frequency_df.columns[allelic_frequency_df.columns.str.contains(alternate_allele)].tolist()[0]

    # Create a list of colors for each popID based on the region
    colors = allelic_frequency_df['region'].map(region_to_color)

    ## plot the bar plot
    ax = allelic_frequency_df.plot(x="popID", y=column_name, kind="bar", color=colors, legend=False)
    ax.set_title(row["gene_name"] + " " + row["seq_name"] + " " + str(row["location"]))

    ## add legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=region_to_color[region]) for region in region_to_color]

    ax.legend(handles, region_to_color.keys(), loc="upper right")
    ax.set_xlabel("Population ID")

    plt.savefig("/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/"
                "compare_landscape_and_outlier_analysis/compare_candidate_allel_alt_allele_freq_over_samples/"
                "alt_allel_freq_distribution_barplot/allele_freq_" + row["gene_name"] + "_" + row["seq_name"] + "_" + str(row["location"]) + ".pdf")
    plt.close()


