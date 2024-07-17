import pandas as pd
from sas7bdat import SAS7BDAT

file_dir = "/Users/ericliao/Desktop/WNV_project_files/OSF_Storage_US_disease_data/wnvvetcounty.sas7bdat"

# Open the SAS file and read the data into a pandas dataframe
with SAS7BDAT(file_dir) as file:
    df = file.to_data_frame()

# Write the dataframe to a CSV file
df.to_csv(file_dir.split(".", 2)[0] + ".csv", index=False)