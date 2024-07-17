### plot disease distribution by 2019 for each county based on coordinates and west nile virus cases onto a unnites states map

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


# read disease data file
cdc_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/monthly/cdc_sum_organism_all_with_phylodiversity.csv")

## agrragate the data by year, FIPS, Latitude and Longitude
cdc_df = cdc_df.groupby(['Year', 'FIPS', 'Latitude', 'Longitude']).agg({'Total_Organism_WNV_Count': 'sum'}).reset_index()

## for loop the list of year to create a plot for each year
year_list = cdc_df["Year"].unique()

for year in year_list:
    ## only 2019 data
    cdc_df_year = cdc_df[cdc_df["Year"] == year]

    # Create a figure and basemap object
    plt.figure(figsize=(12, 9))
    m = Basemap(projection='lcc', resolution='l',
                lat_0=38, lon_0=-97,
                width=5000000, height=3000000)
    # m.shadedrelief()
    m.drawcoastlines(color='gray')
    m.drawcountries(color='gray')
    m.drawstates(color='gray')

    # Plot data points
    x, y = m(cdc_df_year['Longitude'].values, cdc_df_year['Latitude'].values)

    # plot the data as circles where the number of cases is the size of the circle and use a colormap to show the number of cases
    m.scatter(x, y, s=cdc_df_year['Total_Organism_WNV_Count'].values * 5, c=cdc_df_year['Total_Organism_WNV_Count'].values, cmap='plasma', alpha=0.75, zorder=5)

    # Add a legend
    plt.colorbar()

    plt.title('Non-Human West Nile Virus cases in ' + str(year))

    # save the plot
    plt.savefig('/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/yearly/plot/nonhuman_wnv_' + str(year) + '.png')







