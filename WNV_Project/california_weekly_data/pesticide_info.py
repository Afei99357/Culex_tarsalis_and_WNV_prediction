import pandas as pd

# read the csv file
data_2004 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/pesticide/california_2004_pesticide.txt", sep="\t")
data_2005 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/pesticide/california_2005_pesticide.txt", sep="\t")
data_2006 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/pesticide/california_2006_pesticide.txt", sep="\t")
data_2010 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/pesticide/california_2010_pesticide.txt", sep="\t")
data_2011 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/pesticide/california_2011_pesticide.txt", sep="\t")
data_2012 = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/pesticide/california_2012_pesticide.txt", sep="\t")

data_cali = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
                        "add_0_for_no_wnv/cali_week_wnnd_wnf_weather_shift.csv", index_col=False)

# concat 2011 and 2012 data
data = pd.concat([data_2004, data_2005, data_2006, data_2010, data_2011, data_2012], axis=0)

# convert date to datetime
data['DATE'] = pd.to_datetime(data['DATE'])

data.pop("YEAR")

# break the date into year, month, and day
data['Year'] = pd.DatetimeIndex(data['DATE']).year
data['Month'] = pd.DatetimeIndex(data['DATE']).month
data['Day'] = pd.DatetimeIndex(data['DATE']).day

# keep certain columns
data = data[['Year', 'Month', 'COUNTY_NAME']]

# group by year, month, and county name, and count the number of rows, add a columns called "pesticide_count"
pesticide_count = data.groupby(['Year', 'Month', 'COUNTY_NAME']).size().reset_index(name='pesticide_count')

# convert the county name to upper case
pesticide_count['COUNTY_NAME'] = pesticide_count['COUNTY_NAME'].str.upper()
pesticide_count['COUNTY_NAME'] = pesticide_count['COUNTY_NAME'].str.strip()
data_cali['County'] = data_cali['County'].str.upper()

## merge the pesticide_count with the data_cali
data_cali = pd.merge(data_cali, pesticide_count, how='left', left_on=['Year', 'Month', 'County'], right_on=['Year', 'Month', 'COUNTY_NAME'])

# fill the nan with 0 for pesticide_count
data_cali['pesticide_count'] = data_cali['pesticide_count'].fillna(0)

data_cali.pop("COUNTY_NAME")

# create a date column by combine year, month and day as 1
# data_cali['Date'] = data_cali['Year'].astype(str) + "-" + data_cali['Month'].astype(str) + "-" + '1'

# create a column of Human_WNND_Rate by using the formula: Human_WNND_Rate = Human_WNND_Count / Population
data_cali['Human_WNND_Rate'] = data_cali['Human_WNND_Count'] / data_cali['Population']

# save the data
data_cali.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
                 "add_0_for_no_wnv/cali_week_wnnd_wnf_all_features.csv")
