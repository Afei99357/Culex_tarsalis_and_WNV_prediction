### plot disease distribution by each year for each county based on coordinates and west nile virus cases onto a unnites states map

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


# read disease data file
cdc_df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/cdc_human_1999_to_2023/WNV_human_and_non-human_annual_by_county_all_years_impute_missing.csv")

## round the Reported_human_cases column to integer
cdc_df["Reported_human_cases"] = cdc_df["Reported_human_cases"].round().astype(int)

## for loop the list of year to create a plot for each year
year_list = cdc_df["Year"].unique()

for year in year_list:
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
    m.scatter(x, y, s=cdc_df_year['Reported_human_cases'].values * 2, c=cdc_df_year['Reported_human_cases'].values, cmap='plasma', alpha=0.75)

    # ## set the lim of the colorbar
    plt.clim(0, 800)

    # Add a colorbar and title
    plt.colorbar()

    plt.title('Human West Nile Virus Cases in ' + str(year))

    # save the plot
    plt.savefig('/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/yearly/plot/human_wnnv_' + str(year) + '.png')







