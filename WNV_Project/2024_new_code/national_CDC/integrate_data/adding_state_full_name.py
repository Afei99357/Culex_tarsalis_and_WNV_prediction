import pandas as pd
import os

## store all the csv files
csv_list = [f for f in os.listdir("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/"
                                  "cdc_human_1999_to_2023/original_download_files/") if f.endswith(".csv")]


## based on the State_abbrev, add the full name of the state
state_full_name = {
    'AL': 'Alabama',
    'AK': 'Alaska',
    'AZ': 'Arizona',
    'AR': 'Arkansas',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'IA': 'Iowa',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'ME': 'Maine',
    'MD': 'Maryland',
    'MA': 'Massachusetts',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MS': 'Mississippi',
    'MO': 'Missouri',
    'MT': 'Montana',
    'NE': 'Nebraska',
    'NV': 'Nevada',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NY': 'New York',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VT': 'Vermont',
    'VA': 'Virginia',
    'WA': 'Washington',
    'WV': 'West Virginia',
    'WI': 'Wisconsin',
    'WY': 'Wyoming'
}

## create a list to store the dataframes
dfs = []

## for loop each file and combine the data
for file in csv_list:
    df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/"
                     "cdc_human_1999_to_2023/original_download_files/" + file, index_col=False, header=0)

    # separate the first column by comma to two columns. State_abbrev and County
    df[['State_abbrev', 'County']] = df['FullGeoName'].str.split(',', 1, expand=True)
    df['State'] = df['State_abbrev'].map(state_full_name)

    # drop the State_abbrev column
    df = df.drop(columns=["State_abbrev", "FullGeoName"])

    ## append the dataframe to the list
    dfs.append(df)

## concatenate the dataframes
human_df = pd.concat(dfs, axis=0)

## if the State is Louisiana, if the County has parish, replace it with empty string
human_df.loc[human_df["State"] == "Louisiana", "County"] = human_df.loc[human_df["State"] == "Louisiana", "County"].str.replace(" Parish", "")

## if any County Names begins with "St ", replace it with "St. "
human_df["County"] = human_df["County"].str.replace("St ", "St. ")

## convert County and State to lower case
human_df["County"] = human_df["County"].str.lower()
human_df["State"] = human_df["State"].str.lower()

## remove any leading and trailing white spaces on county and state
human_df["County"] = human_df["County"].str.strip()
human_df["State"] = human_df["State"].str.strip()

## read the file with FIPS code
disease_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/monthly/"
                       "cdc_sum_organism_all_with_phylodiversity.csv", index_col=0)

## only keep columns that are needed, FIPS, County, State, Latitude, Longitude
disease_df = disease_df[["FIPS", "County", "State", "Latitude", "Longitude"]]

## drop the duplicated rows based on FIPS
disease_df = disease_df.drop_duplicates(subset=["FIPS"])

## based on the County and State, adding the FIPS, Latutude and Longitude to the human_df
human_df = human_df.merge(disease_df, on=["County", "State"], how="left")

## save the dataframe to a csv file
human_df.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/cdc_human_1999_to_2023/"
                "West_Nile_virus_human_and_non-human_activity_by_county_1999_to_2023_CDC_new.csv", index=False)




