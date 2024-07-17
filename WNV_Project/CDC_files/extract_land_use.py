import rasterio
from rasterio.transform import from_origin
import pandas as pd

df = pd. read_csv("/Users/ericliao/Desktop/WNV_project_files/CDC-DATA-2023/monthly/"
                  "cdc_sum_organism_all_with_phylodiversity.csv", index_col=0)

# drop row if the coordinate is nan
df = df.dropna(subset=["Latitude", "Longitude"])

# get unique pair of coordinates
unique_coordinates = df[["Latitude", "Longitude"]].drop_duplicates()

# reindex the unique_coordinates
unique_coordinates = unique_coordinates.reset_index(drop=True)

layer_filenames = [f"/Users/ericliao/Desktop/WNV_project_files/weather_and_land_use/consensus_land_cover_data/" \
                       f"consensus_full_class_{i}.tif" for i in range(1, 2)]

# loop over each file, and pass through the pixel value at the location
for filename in layer_filenames:
    print("Processing file", filename)
    with rasterio.open(filename) as src:
        # ###
        # latitude = 34.5
        # longitude = -80.8
        #
        # # Convert geographic coordinates to pixel coordinates
        # resolution = 1  # 1 km per pixel at the equator
        # transform = from_origin(-180, 90, resolution, resolution)  # Assuming the data extent is from 90 N - 56 S and from 180W - 180E
        # col_index, row_index = rasterio.transform.rowcol(transform, longitude, latitude)
        #
        # # Read pixel value at the location from Band 1
        # pixel_value = src.read(1, window=((row_index - 1, row_index + 1), (col_index - 1, col_index + 1)))
        #
        # # Access the pixel value
        # band1_value = pixel_value[0][0]
        #
        # # Print the extracted feature values from Band 1
        # print("Feature value from Band 1:", band1_value)
        #
        # ###


        for index, row in unique_coordinates.iterrows():
            print("Processing row", index)
            latitude = row["Latitude"]
            longitude = row["Longitude"]

            # Convert geographic coordinates to pixel coordinates
            resolution = 1  # 1 km per pixel at the equator
            transform = from_origin(-180, 90, resolution, resolution)  # Assuming the data extent is from 90 N - 56 S and from 180W - 180E
            col_index, row_index = rasterio.transform.rowcol(transform, longitude, latitude)

            print("Pixel coordinates:", col_index, row_index)

            # Find the nearest pixel coordinates
            col_index = int(round(col_index))
            row_index = int(round(row_index))
            col_index = max(0, min(col_index, src.width - 1))
            row_index = max(0, min(row_index, src.height - 1))

            # Read pixel value at the location
            pixel_value = src.read(1, window=((row_index, row_index + 1), (col_index, col_index + 1)))[0][0]

            # Scale the pixel value to percentage
            # percentage_value = pixel_value * 100 / 255  # Assuming 8-bit unsigned values

            # Print the extracted feature values
            print("Feature values:", pixel_value)

