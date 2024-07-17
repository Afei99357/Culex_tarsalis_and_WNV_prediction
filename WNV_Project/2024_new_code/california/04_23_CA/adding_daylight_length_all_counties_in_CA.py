import pandas as pd
import numpy as np
from astral.sun import sun
from astral import LocationInfo
from datetime import datetime
import calendar

# Load the dataset into a Pandas DataFrame
df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/CA_human_data_2004_to_2023_final_no_impute_0.csv",
                   index_col=False,
                   header=0)

def calculate_daylight_length(latitude, longitude, year, month):
    location = LocationInfo(latitude=latitude, longitude=longitude)
    total_daylight = 0
    days_in_month = calendar.monthrange(year, month)[1]

    for day in range(1, days_in_month + 1):
        date = datetime(year, month, day)
        s = sun(location.observer, date=date)
        daylight_length = (s['sunset'] - s['sunrise']).seconds / 3600  # Daylight length in hours
        total_daylight += daylight_length

    average_daylight = total_daylight / days_in_month
    return average_daylight

# Apply the function to each row in the DataFrame
df['Average_Daylight_Hours'] = df.apply(lambda row: calculate_daylight_length(row['Latitude'], row['Longitude'], row['Year'], row['Month']), axis=1)

## save the new dataset to a csv file
df.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/CA_human_data_2004_to_2023_final_no_impute_0_adding_daylight.csv", index=False)
