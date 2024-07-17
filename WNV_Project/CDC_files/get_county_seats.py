import geopandas as gpd
from geopy.geocoders import Nominatim

# Read the GNIS dataset for county seats in the United States
gnis_data = gpd.read_file('https://geonames.usgs.gov/docs/stategaz/NationalFile_20210301.zip')

# Filter the dataset to include only county seats
county_seats = gnis_data[gnis_data['FEATURE_CLASS'] == 'Seat of government']
