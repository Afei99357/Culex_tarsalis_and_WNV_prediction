#### go through each file in the folder /Users/ericliao/Desktop/WNV_project_files/weather_and_land_use/california_hourly_data and if the name contains illinois, then change it to california
import os
import glob

# get the file list
file_list = glob.glob("/Users/ericliao/Desktop/WNV_project_files/weather_and_land_use/california_hourly_data/*.nc")

# for loop
for file in file_list:
    # get the file name
    file_name = os.path.basename(file)

    # if the file name contains illinois, then change it to california
    if "illinois" in file_name:
        # change the file name
        os.rename(file, file.replace("illinois", "california"))

        # print the file name
        print(file_name)

