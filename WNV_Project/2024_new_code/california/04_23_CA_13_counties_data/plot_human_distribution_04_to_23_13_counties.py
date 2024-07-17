### plot disease distribution by each year for each county based on coordinates and west nile virus cases onto a unnites states map

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/"
                          "CA_13_county_dataset/CA_13_counties_04_23_impute_0.csv",
                   index_col=False,
                   header=0)

## for loop the list of year to create a plot for each year
year_list = data["Year"].unique()

## only keep the columns: Year, Month, Latitude, Longitude and Human_Disease_Count
data = data[["Year", "Month", "Latitude", "Longitude", "Human_Disease_Count"]]

## group by Year, Month, Latitude and Longitude and sum the Human_Disease_Count
data = data.groupby(["Year", "Latitude", "Longitude"]).sum().reset_index()

## Keep longitude and latitude with 2 decimal points
data["Latitude"] = data["Latitude"].round(2)
data["Longitude"] = data["Longitude"].round(2)

for year in year_list:
    ca_df_year = data[data["Year"] == year]

    # Create a figure and basemap object
    plt.figure(figsize=(12, 9))
    m = Basemap(projection='merc', resolution='h',
                llcrnrlon=-125, llcrnrlat=32,
                urcrnrlon=-114, urcrnrlat=42)
    m.drawcoastlines(color='gray')
    m.drawcountries(color='gray')
    m.drawstates(color='gray')
    m.drawcounties(color='gray')

    # Plot data points
    x, y = m(ca_df_year['Longitude'].values, ca_df_year['Latitude'].values)

    # plot the data as circles where the number of cases is the size of the circle and use a colormap to show the number of cases
    m.scatter(x, y, s=ca_df_year['Human_Disease_Count'].values * 2, c=ca_df_year['Human_Disease_Count'].values, cmap='RdYlGn', alpha=0.75)

    ## set the lim of the colorbar
    plt.clim(0, 150)

    # Add a legend
    plt.colorbar()

    plt.title('Human West Nile Virus Cases in ' + str(year))

    # save the plot
    plt.savefig('/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/result/plots/yearly_cases_distribution/human_wnv_' + str(year) + '.png')







