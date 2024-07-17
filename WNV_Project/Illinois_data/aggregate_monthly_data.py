import pandas as pd

# load the data
df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/"
                 "non_human_data_2002_2017_with_species_modified.csv", index_col=0)

# # drop the rows where the Municipality, Date_Collected, Species_modified is empty
df = df.dropna(subset=["Municipality", "Date_Collected", "Species_Modified"])

# Define datetime formats to try
date_formats = ["%d-%b-%y", "%m/%d/%y"]

# Convert "Date" column to datetime format
new_dates = []
for date_str in df["Date_Collected"]:
    date_obj = None
    for fmt in date_formats:
        try:
            date_obj = pd.to_datetime(date_str, format=fmt)
            break
        except ValueError:
            pass
    new_dates.append(date_obj)

df["Date_Collected"] = new_dates

# # drop the rows where the Date_Collected is NaT
df = df.dropna(subset=["Date_Collected"])

# # add a column for Year
df["Year"] = df["Date_Collected"].dt.year
# # add a column for Month
df["Month"] = df["Date_Collected"].dt.month

# # groupby data based on Year, Month, Municipality, Latitude, Longitude and Species_modified
df = df.groupby(["Year", "Month", "Municipality", "Latitude", "Longitude", "Species_Modified"]).size().reset_index(name="Count")

# pivot the data based on Year, Month, Municipality, Latitude, Longitude and Species_modified
df = df.pivot_table(index=["Year", "Month", "Municipality", "Latitude", "Longitude"], columns="Species_Modified", values="Count").reset_index()

# # save the data
df.to_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/non_human_data_2002_2017_monthly_aggregated.csv")
