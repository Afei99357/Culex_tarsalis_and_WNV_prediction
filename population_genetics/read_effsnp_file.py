import pandas as pd

use_cols2 = [0, 1, 7]

# read a vcf file
df_vcf = pd.read_csv(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/original_vcf/bi_20missing_filtSNP_maf_005.recode.vcf",
    header=0,
    index_col=False,
    sep="\t",
    skiprows=9,
)
df_vcf
