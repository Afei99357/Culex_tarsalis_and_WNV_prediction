from uszipcode import SearchEngine
import pandas as pd
from multiprocessing import Pool
import traceback

# based on the municipality and state
# get the latitude and longitude
state = "Illinois"
search = SearchEngine()

def get_coordinates(row):
    index, row = row
    print(index)
    # if the municipality is not empty
    if pd.isna(row["Municipality"]):
        # add empty values to the lists
        return "", ""

    # try catch block to handle the error
    try:

        result = search.by_city_and_state(row["Municipality"], state)
    except:
        print("Error for municipality:", row["Municipality"])
        traceback.print_exc()
        # add empty values to the lists
        return "", ""

    if result:
        return result[0].lat, result[0].lng
    else:
        # add empty values to the lists
        return "", ""
        print("No results found for municipality:", row["Municipality"])

if __name__ == '__main__':
    # load the municipality data
    municipality_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/"
                                  "non_human_data_2002_2017_fill_missing_count.csv", index_col=0)

    # create two columns for latitude and longitude
    municipality_df["Latitude"] = ""
    municipality_df["Longitude"] = ""

    # get the list of municipality
    municipality_list = municipality_df["Municipality"].tolist()

    ## execute every row in the dataframe in different processes
    with Pool() as p:
        latitudes, longitudes = zip(*p.map(get_coordinates, municipality_df.iterrows()))

    # add the lists to the dataframe
    municipality_df["Latitude"] = latitudes
    municipality_df["Longitude"] = longitudes

    # save the data
    municipality_df.to_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/"
                           "non_human_data_2002_2017_with_coordinates.csv")