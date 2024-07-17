## rename the file to the new name with uniform format

import os

## get all the pdf files in the directory
pdf_files = []

for file in os.listdir("/Users/ericliao/Desktop/WNV_project_files/WNV/california/all_arbobulletins/original_files"):
    pdf_files.append(os.path.join("/Users/ericliao/Desktop/WNV_project_files/WNV/california/all_arbobulletins/original_files", file))

## rename the file to the new name with uniform format

## get year list
year_list = [str(i) for i in range(2004, 2024)]

## for each file, remove "#" if there is any, and rename the file to the new name
for pdf_file in pdf_files:
    name = pdf_file.split("/")[-1].split("_")[0]
    ## get the year from the file name if there are four digits
    year = pdf_file.split("/")[-1].split("_")[1]
    ## get the file_no from the file name, remove the "#" or - if there is any
    file_no = pdf_file.split("/")[-1].split("_")[2].replace("#", "")
    ## if there is -, only keep the first part
    file_no = file_no.split("-")[0]
    ## if the file number is one digit, add a "0" in front of the number
    if len(file_no) == 1:
        file_no = "0" + file_no
    ## only keep the first two digits of the file_no
    file_no = file_no[:2]

    ## get the new file name
    new_file_name = f"{name}_{year}_{file_no}.pdf"
    ## rename the file
    os.rename(pdf_file, os.path.join("/Users/ericliao/Desktop/WNV_project_files/WNV/california/all_arbobulletins/rename_files_copy", new_file_name))
