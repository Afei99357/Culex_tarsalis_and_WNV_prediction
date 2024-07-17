import pandas as pd

# read interproscan file
df_interproscan = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/original_vcf/"
    "Culex-tarsalis-v1.0.a1.5d6405151b078-interproscan.tab",
    header=8,
    index_col=False,
    sep="\t",
)

## read annotation gff3 file
df_gff3 = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/original_vcf/"
    "Culex-tarsalis-v1.0.a1-merged-2019-08-30-4-45-01.gff3",
    header=None,
    index_col=False,
    sep="\t",
    skiprows=3
)

## get the subset of df_gff3 where the 3rd column is gene
df_gff3_gene = df_gff3[df_gff3[2] == "gene"]

## create an empty df_fCHr to store the GeneID as GID and Chromosome as CHROMOSOME
df_fCHr = pd.DataFrame(columns=["GID", "CHROMOSOME"])

## create an dataarray to store the GeneID as GID and Chromosome as CHROMOSOME
data = {"GID": [], "CHROMOSOME": []}

## for each row in df_gff3_gene, get the GeneID and Chromosome and append to df_fCHr
## frist column is the Chromosome, the 9th column contains the GeneID. the 9th columns separate by ;
## the first values in the 9th column is ID=GeneID, extract the GeneID
for index, row in df_gff3_gene.iterrows():
    gene_id = row[8].split(";")[0].split("=")[1]
    chromosome = row[0]
    ## add the GeneID and Chromosome to the dataarray
    data["GID"].append(gene_id)
    data["CHROMOSOME"].append(chromosome)

## add the dataarray to df_fCHr
df_fCHr = pd.DataFrame(data)

## get subset of df_interproscan where the Source column is PANTHER
df_interproscan_panther = df_interproscan[df_interproscan["Source"] == "PANTHER"]

## create a new dataframe to store the GeneID as GID, Source Term as SYMBOL, Source Description as GENENAME
df_modify = pd.DataFrame(columns=["GID", "SYMBOL", "GENENAME", "GO"])

## create an dataarray to store the GeneID as GID, Source Term as SYMBOL, Source Description as GENENAME
data = {"GID": [], "SYMBOL": [], "GENENAME": [], "GO": []}

## from the df_interproscan_panther, extract the GeneID as GID from the Name where the name is in the format of 20471-GeneID.m01,
## extract the Source Term save as SYMBOL, extract the Source Description as GENENAME
for index, row in df_interproscan_panther.iterrows():
    gene_id = row["Name"].split("-")[1].split(".m")[0]
    symbol = row["Source Term"]
    gene_name = row["Source Description"]
    go = row["GO"]
    ## add the GeneID, Source Term and Source Description to the dataarray
    data["GID"].append(gene_id)
    data["SYMBOL"].append(symbol)
    data["GENENAME"].append(gene_name)
    data["GO"].append(go)

## add the dataarray to df_fSym
df_modify = pd.DataFrame(data)

## remove the duplicate rows in df_fSym
df_modify = df_modify.drop_duplicates()

## if there is more than 1 row has the same GID, keep the row where the SYMBOL contains no : in it
df_modify = df_modify.sort_values(by=["GID", "SYMBOL"])
df_modify = df_modify.drop_duplicates(subset=["GID"], keep="first")


## subset df_modify and only keep GID, SYMBOL AND GENENAME
df_fSym = df_modify[["GID", "SYMBOL", "GENENAME"]]

## go df_modify and only keep GID, GO
df_fGO = df_modify[["GID", "GO"]]

## add new colunn to df_fGO, column name is EVIDENCE, value is IEA
df_fGO["EVIDENCE"] = "IEA"

## remove the nan value in df_fGO
df_fGO = df_fGO.dropna()

## if in the GO column, there are multiple GO terms, split the GO terms by , and create new rows for each GO term
## and copy the GID and EVIDENCE to the new rows
df_fGO = df_fGO.assign(GO=df_fGO["GO"].str.split(", ")).explode("GO")
df_fGO = df_fGO.assign(EVIDENCE=df_fGO["EVIDENCE"].str.split(",")).explode("EVIDENCE")

## drop the rows in fCHR where the GID is not in df_fSym
df_fCHr = df_fCHr[df_fCHr["GID"].isin(df_fSym["GID"])]

## output df_fCHR, df_fSYm and df_fGO as csv files
df_fCHr.to_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/"
    "compare_landscape_and_outlier_analysis/cluster_go_term/fCHR.csv",
    index=False)

df_fSym.to_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/"
    "compare_landscape_and_outlier_analysis/cluster_go_term/fSym.csv",
    index=False)

df_fGO.to_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/"
    "compare_landscape_and_outlier_analysis/cluster_go_term/fGO.csv",
    index=False)


## check the number of unique GID only in df_fCHr, and only in df_fSym, and only in df_fGO
## and the number of GID in df_fCHr, df_fSym, df_fGO
print(len(df_fCHr["GID"].unique()))
print(len(df_fSym["GID"].unique()))
print(len(df_fGO["GID"].unique()))

## plot the venn diagram
from venn import venn
import matplotlib.pyplot as plt

venn({"fCHr": set(df_fCHr["GID"]), "fSym": set(df_fSym["GID"]), "fGO": set(df_fGO["GID"])})
plt.show()










