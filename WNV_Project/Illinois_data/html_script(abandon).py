import requests
import time
import pandas as pd
from lxml import html

import urllib.request
from bs4 import BeautifulSoup

# page_url = "envhealth/wnvsurveillance_data_02.htm"
# url = "http://www.idph.state.il.us/" + page_url
# response = urllib.request.urlopen(url)
# html_content = response.read().decode()

# for year in ["05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17"]:
year = "06"
human_tables = []
animal_tables = []

text = requests.get(f"http://www.idph.state.il.us/envhealth/wnvsurveillance_data_{year}.htm").text

root = html.fromstring(text)

for link in root.xpath("//tr/td[@width='98']/font/a"):
    print(f"Loading {link.text}")
    page_url = link.get("href")
    # table_page_text = requests.get("http://www.idph.state.il.us/envhealth/" + page_url).text

    response = urllib.request.urlopen("http://www.idph.state.il.us/envhealth/" + page_url)
    html_content = response.read()

    # parse the HTML content using BeautifulSoup
    table_page_text = BeautifulSoup(html_content, 'html.parser')

    table_page_root = html.fromstring(table_page_text)
    tables_in_detail_page = table_page_root.xpath("//td[@width=475]/center//table")
    if len(tables_in_detail_page) == 1:
        # only animals
        animal_table = tables_in_detail_page[0]
        animal_table = html.tostring(animal_table)
        animal_table = pd.read_html(animal_table)
        animal_tables.append(animal_table)
    else:
        # humans and animals
        human_table = tables_in_detail_page[0]
        human_table = html.tostring(human_table)
        human_table = pd.read_html(human_table)
        human_tables.append(human_table)

        animal_table = tables_in_detail_page[1]
        animal_table = html.tostring(animal_table)
        animal_table = pd.read_html(animal_table)
        animal_tables.append(animal_table)
    time.sleep(1)

# combine all the animal tables
animal_df = pd.concat([df[0] for df in animal_tables], axis=0, ignore_index=True)

# combine all the human tables
human_df = pd.concat([df[0] for df in human_tables], axis=0, ignore_index=True)

# save the data
animal_df.to_csv(f"/Users/ericliao/Desktop/WNV_project_files/illinois_data/animal_data_{year}.csv", index=False)
human_df.to_csv(f"/Users/ericliao/Desktop/WNV_project_files/illinois_data/human_data_{year}.csv", index=False)