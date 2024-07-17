from uszipcode import SearchEngine
import pandas as pd

# load the municipality data
municipality_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/"
                              "non_human_data_2002_2017_monthly_aggregated.csv", index_col=False)

# create a county column
municipality_df["County"] = ""

# get the list of municipality
municipality_list = municipality_df["Municipality"].tolist()

# for each municipality, get the county name
search = SearchEngine()

search.by_city_and_state("Barlett", 'Illinois')

for municipality in municipality_list:
    print(municipality)
    # check if the municipality is a valid city name, if it is, get the county name, if not, continue to next
    try:
        if search.by_city_and_state(municipality, 'Illinois') is None:
            continue

        result = search.by_city_and_state(municipality, 'Illinois')
        if result:
            county_name = result[0].county
            # fill in the county name
            municipality_df.loc[municipality_df["Municipality"] == municipality, "County"] = county_name
        else:
            print("No results found for municipality:", municipality)
    except:
        print("Error for municipality:", municipality)
        continue

# remove "County" from the county name
municipality_df["County"] = municipality_df["County"].str.replace(" County", "")

# lower case the values in County and Municipality
municipality_df["County"] = municipality_df["County"].str.lower()
municipality_df["Municipality"] = municipality_df["Municipality"].str.lower()
# strip the white space in County and Municipality
municipality_df["County"] = municipality_df["County"].str.strip()
municipality_df["Municipality"] = municipality_df["Municipality"].str.strip()

# # create a dictionary where the key is municipality and the value is county
municipality_county = municipality_df[["County", "Municipality"]].drop_duplicates()
municipality_county = municipality_county.set_index("Municipality").to_dict()["County"]

# save the data
municipality_df.to_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/non_human_data_2002_2017_monthly_aggregated_by_county_2.csv")
#
# # import the data has missing county
# df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/non_human_data_2002_2017_fill_missing_county.csv", index_col=0)
#
# # for original data, if the County is missing, fill in the County name based on the Municipality if it exists in the dictionary's keys
# def fill_county(row):
#     if pd.isna(row["County"]):
#         if row["Municipality"] in municipality_county.keys():
#             row["County"] = municipality_county[row["Municipality"]]
#     return row
#
# df = df.apply(fill_county, axis=1)
#
# # save the data
# df.to_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/non_human_data_2002_2017_fill_missing_count_2.csv")

