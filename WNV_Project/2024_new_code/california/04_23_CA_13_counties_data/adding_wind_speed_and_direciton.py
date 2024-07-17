import pandas as pd
import numpy as np
from astral.sun import sun
from astral import LocationInfo
from datetime import datetime
import calendar

# Load the dataset into a Pandas DataFrame
df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/CA_13_counties_04_23_no_impute_daylight.csv",
                   index_col=False,
                   header=0)

def calculate_wind_speed_direction(u10, v10):
    wind_speed = np.sqrt(u10 ** 2 + v10 ** 2)
    wind_direction = (np.arctan2(u10, v10) * 180 / np.pi) % 360
    return wind_speed, wind_direction

# Apply the function to each row in the DataFrame
df['wind_speed_1m_shift'], df["wind_direction_1m_shift"] = zip(*df.apply(lambda row: calculate_wind_speed_direction(row['u10_1m_shift'], row['v10_1m_shift']), axis=1))
## save the new dataset to a csv file
df.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/CA_13_counties_04_23_no_impute_wind.csv", index=False)
