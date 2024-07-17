import datetime
import pandas as pd
import numpy as np

# # import the data

data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/Veterinary_Numerator.csv", index_col=False)

# # get the data only horses
horse_data = data[(data["SpeciesName"] == "Equine") & (data["State"] == "CA")]

# # add date column by combine Year and Week
horse_data["Date"] = horse_data["Year"].astype(str) + "-" + horse_data["Week"].astype(str)

# # convert "Date" column to datetime
# horse_data["Date"] = pd.to_datetime(horse_data["Date"], format="%Y-%W")

# # covert the "Date" column to the first day of the week
horse_data["Date"] = horse_data["Date"] - pd.to_timedelta(horse_data["Date"].dt.dayofweek, unit='d')
# datetime.datetime.strptime(horse_data["Date"], "%Y-%W-%w")

# # drop the "Year" and "Week" columns
horse_data = horse_data.drop(["Year", "Week"], axis=1)


