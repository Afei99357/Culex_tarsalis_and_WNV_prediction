from uszipcode import SearchEngine
import pandas as pd
from multiprocessing import Pool
import traceback

# based on the municipality and state
# get the latitude and longitude
search = SearchEngine()

def get_coordinates(row):
    index, row = row
    print(index)
    # if the municipality is not empty
    if pd.isna(row["County_Seat"]):
        # add empty values to the lists
        return "", ""

    # try catch block to handle the error
    try:
        result = search.by_city_and_state(row["County_Seat"], row["State"])
    except:
        print("Error for County_Seat:", row["County_Seat"])
        traceback.print_exc()
        # add empty values to the lists
        return "", ""

    if result:
        return result[0].lat, result[0].lng
    else:
        # add empty values to the lists
        return "", ""
        print("No results found for municipality:", row["County_Seat"])

if __name__ == '__main__':
    # load the municipality data
    cdc_sum_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/cdc_sum_organism.csv",
                             index_col=0)

    # create two columns for latitude and longitude
    cdc_sum_df["Latitude"] = ""
    cdc_sum_df["Longitude"] = ""

    ## execute every row in the dataframe in different processes
    with Pool() as p:
        latitudes, longitudes = zip(*p.map(get_coordinates, cdc_sum_df.iterrows()))

    # add the lists to the dataframe
    cdc_sum_df["Latitude"] = latitudes
    cdc_sum_df["Longitude"] = longitudes

    # save the data
    cdc_sum_df.to_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/"
                      "cdc_sum_organism_with_coordinates.csv")