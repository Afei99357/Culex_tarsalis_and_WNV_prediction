import datetime

import cdsapi

c = cdsapi.Client()

# write a for loop to loop through all the months and years in 2017, 2018, 2019, 2020, 2021, 2022

# # create a list of years for illinois
# year_list = ["2017", "2018", "2019", "2020", "2021", "2022"]

# # create a list of years for california
year_list = ["2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013",
             "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]

# # create a list of months
month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

# # create a list of variables
variable_list = [
    # "skin_reservoir_content",
    # "total_precipitation",
    # '10m_u_component_of_wind',
    # '10m_v_component_of_wind',
    '2m_temperature',
    # 'leaf_area_index_high_vegetation',
    # 'leaf_area_index_low_vegetation',
    # 'skin_reservoir_content',
    # 'snow_depth',
    # 'surface_runoff',

]

# # loop through each variable, year and month
for variable in variable_list:
    for year in year_list:
        for month in month_list:

            # print notification for what time started to retrieve data
            print(f"Started retrieving {variable} for " + year + "-" + month + " at " + str(datetime.datetime.now()))

            # # # retrieve data from CDS
            c.retrieve(
                'reanalysis-era5-land',
                {
                    'variable': [
                        variable,
                    ],
                    'year': year,
                    'month': month,
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00',
                        '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],

                    # # #for illinois
                    # 'area': [43, -92, 36, -86, ],

                    #for California
                    'area': [49.4, -124.7, 25.1, -66.9],
                    # between 25.1 N
                    # and 49.4 N and the longitude range is between 66.9 W and 124.7 W.

                    'format': 'netcdf',
                },
                '/Users/ericliao/Desktop/WNV_project_files/weather_and_land_use/hourly_data_national/'
                f'era5_land_US_county_{variable}_' + year + '_' + month + '.nc')

            # print notification for what time finished retrieving data
            print(f"Finished retrieving {variable} for " + year + "-" + month + " at " + str(datetime.datetime.now()))