## combine all the pdf file in the same year text as one text file
#
import pdfplumber
import os

## get all the pdf files in the directory
pdf_files = []

for file in os.listdir("/Users/ericliao/Desktop/WNV_project_files/WNV/california/all_arbobulletins/rename_files_copy"):
    pdf_files.append(os.path.join("/Users/ericliao/Desktop/WNV_project_files/WNV/california/all_arbobulletins/rename_files_copy", file))

## get year list
year_list = [str(i) for i in range(2004, 2024)]

## for each year, combine all the first two pages of each pdf files as one text file
for year in year_list:
    ## get all the pdf files in the directory with file name containing the year
    pdf_files_year = []
    for pdf_file in pdf_files:
        if year in pdf_file:
            pdf_files_year.append(pdf_file)

    ## sort the pdf files based on the ascending order of the number in the file name
    pdf_files_year.sort()

    ## combine all the text in the first two pages of each pdf file as one text file
    text_all = ""
    for pdf_file in pdf_files_year:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages[:1]:
                text = page.extract_text()
                text_all += text

    ## write the text to a text file
    with open(f"/Users/ericliao/Desktop/WNV_project_files/WNV/california/all_arbobulletins/{year}_2_pages.txt", "w") as f:
        f.write(text_all)
