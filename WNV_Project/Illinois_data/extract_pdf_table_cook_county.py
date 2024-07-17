from tabula.io import read_pdf
import pandas as pd

pdf_path = '/Users/ericliao/Desktop/Cook11.pdf'

# Get the total number of pages in the PDF file
total_pages = len(read_pdf(pdf_path, pages="all"))

print(total_pages)

# Initialize an empty list to store all the extracted tables
tables = []

# Loop through all the pages and extract the tables
for page in range(1, total_pages + 1):
    # Extract table from current page and append to list of tables
    table = read_pdf(pdf_path, pages=page, multiple_tables=False, pandas_options={'header': None})
    # create a dataframe from the list, tab for each column and new line for each row
    df = pd.DataFrame(table[0].values.tolist())
    tables.append(df)

# Concatenate all the tables into a single DataFrame
all_tables = pd.concat(tables, ignore_index=True)

# Print the final DataFrame containing all the extracted tables
all_tables.to_csv("/Users/ericliao/Desktop/cook_11.csv")
