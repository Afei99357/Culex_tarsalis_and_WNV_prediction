import pdfplumber
from subprocess import check_output
import requests
import os

## find all the pdf files in the directory based on the year

year_list = [str(i) for i in range(2004, 2024)]

main_path = "/Users/ericliao/Desktop/WNV_project_files/WNV/california/all_arbobulletins/rename_files"

## get all the pdf files in the directory with file name containing the year
for year in year_list:
    pdf_files = []
    for file in os.listdir(main_path):
        if year in file:
            pdf_files.append(os.path.join(main_path, file))

                ## need to start with the first file, and work through each one by the time order,
                # extract the human cases for each county in the first week, and based on the text of next week, to find out
                # the new cases for each county in the next week. until we finish all the files in one year. then move to next year.
                # and repeat the process until we finish all the files in the directory.

    ## order the pdf files based on the ascending order of the number in the file name
    pdf_files.sort()

        ## iterate through each pdf file
    for pdf_file in pdf_files:
        ## open the pdf file
        ## print file name
        print(pdf_file.split("/")[-1])
        with pdfplumber.open(pdf_file) as pdf:
            ## combine all the text in each page together
            text_all = ""
            for page in pdf.pages:
                text = page.extract_text()
                # This can be useful for pasting to chatgpt for comparison
                # print("Extract the total number of cases from the following pages, or 0 if there are no cases mentioned.\n\n" + text)
                # ollama_out = check_output([
                #     "ollama", "run", "llama3",
                #     "Extract the total number of cases from the following pages, or 0 if there are no cases mentioned.\n\n" + text
                # ])
                text_all += text
            ollama_out = requests.post("http://localhost:11434/api/generate", json={
                "model": "llama3",
                "prompt": "if the text contains non-0 humans disease cases for different county, Tersely extract the number of new " +
                          "cases of mentioned county for the current week. The information should include: year, " +
                          "week, county, number of human cases. ignore the dead birds, mosquito pools and sentinel chickens cases number. Respond using JSON\n\n" + text_all,
                "stream": False,
                "format": "json"
            }).json()["response"]
            print(ollama_out)


