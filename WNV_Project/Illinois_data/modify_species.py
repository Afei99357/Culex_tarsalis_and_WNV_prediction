from typing import List, Any

import pandas as pd

# load the data
df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/non_human_data_2002_2017_with_coordinates.csv",
                 index_col=0)

# add a column for species_modified
df["Species_modified"] = ""

# get the list of species
species_list = df["Species"].tolist()

species_modified_list = []

# create a new list to store the modified species
for i in species_list:
    # if the species is not empty
    if pd.isna(i):
        # add empty values to the lists
        species_modified_list.append("")
        continue

    if i.lower() == 'equine' or i.lower() == 'horse':
        species_modified_list.append('Horse')
        continue

    if i.lower() == 'mosquito' or i.lower() == 'mosquitoes':
        species_modified_list.append('Mosquitoes')
        continue

    else:
        species_modified_list.append("Bird")
        continue

# add the lists to the dataframe
df["Species_modified"] = species_modified_list

# save the data
df.to_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/non_human_data_2002_2017_with_species_modified.csv")

# # groupby data based on Year, Month, Municipality, and Species_modified



