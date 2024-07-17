from geopy.geocoders import Nominatim
import traceback
import pandas as pd
from multiprocessing import Pool

# based on the municipality and state
# get the latitude and longitude
geolocator = Nominatim(user_agent="GEO")

def get_coordinates(row):
    index, row = row
    print(index)
    # if the municipality is not empty
    if pd.isna(row["County_Seat"]):
        # add empty values to the lists
        return "", ""

    # try catch block to handle the error

    city = row['County_Seat']
    state = row['State']

    try:
        result = geolocator.geocode(f"{city}, {state}")
    except:
        # print("Error for County_Seat:", row["County_Seat"], "and State:", row["State"])
        traceback.print_exc()
        # add empty values to the lists
        return "", ""

    if result:
        return result.latitude, result.longitude
        print("success ", result.latitude, result.longitude)
    else:
        # add empty values to the lists
        return "", ""
        print("Coordinates not found.")

if __name__ == '__main__':
    # load the municipality data
    cdc_sum_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/cdc_sum_organism.csv",
                             index_col=0)

    ## execute every row in the dataframe in different processes
    with Pool() as p:
        latitudes, longitudes = zip(*p.map(get_coordinates, cdc_sum_df.iterrows()))

    # add the lists to the dataframe
    cdc_sum_df["Latitude"] = latitudes
    cdc_sum_df["Longitude"] = longitudes

    # save the data
    cdc_sum_df.to_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/"
                      "cdc_sum_organism_with_coordinates_geocoding.csv")