import pdfplumber
from pprint import pprint
import pandas as pd
from tabula import read_pdf

df = read_pdf("/Users/ericliao/Desktop/WNV_project_files/disease_data_weekly_CA/Arbobulletin_2011_#25_3_17.pdf", multiple_tables=True, pages="all")

df.to_csv("/Users/ericliao/Desktop/WNV_project_files/disease_data_weekly_CA/parse_files/file_0.csv", index=False)

# pdf = pdfplumber.open("/Users/ericliao/Desktop/WNV_project_files/disease_data_weekly_CA/Arbobulletin_2011_#25_3_17.pdf")
#
# page_number = 5
#
# page = pdf.pages[page_number].extract_tables({"horizontal_strategy": "text", "min_words_vertical": 1})
#
# num_dic = len(pdf.pages[page_number].extract_tables(dict()))
#
# for i in range(num_dic):
#     row_lists = pdf.pages[page_number].extract_tables(dict())[i]
#     df = pd.DataFrame(row_lists[0:], columns=range(len(row_lists[0])))
#     df.to_csv("/Users/ericliao/Desktop/WNV_project_files/disease_data_weekly_CA/parse_files/file_{}.csv" .format(i),
#               index=False)