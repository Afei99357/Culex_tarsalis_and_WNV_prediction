import pandas as pd

# # load the data
df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/"
                 "non_human_data_2002_2017.csv", index_col=False)

# # lower case the values in County and Municipality
df["County"] = df["County"].str.lower()
df["Municipality"] = df["Municipality"].str.lower()
# # strip the white space in County and Municipality
df["County"] = df["County"].str.strip()
df["Municipality"] = df["Municipality"].str.strip()

# # get column County and Municipality, and get the unique pair stored in a dictionary where the key is municipality
# and the value is county
county_municipality = df[["County", "Municipality"]].drop_duplicates()
county_municipality = county_municipality.set_index("Municipality").to_dict()["County"]

# for original data, if the County is missing, fill in the County name based on the Municipality if it exists in the dictionary
df["County"] = df.apply(lambda x: county_municipality[x["Municipality"]] if pd.isna(x["County"]) else x["County"], axis=1)

df.to_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/non_human_data_2002_2017_fill_missing_county.csv")
