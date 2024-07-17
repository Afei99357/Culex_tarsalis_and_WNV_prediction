import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
import glob

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/"
                   "human/cdc_human_1999_to_2023/WNV_human_and_non_human_yearly_climate_demographic_bird.csv", index_col=0)

## get unique states
states = data["State"].unique()

## agrregate the number of cases for each state for each year
data = data.groupby(["Year", "State"]).agg({"Neuroinvasive_disease_cases": "sum"}).reset_index()

## create a dictionary for the lower case state names and its initial names
state_name_dict = {"alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
                     "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
                        "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
                        "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD", "massachusetts": "MA",
                        "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO", "montana": "MT",
                        "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
                        "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
                        "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
                        "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
                        "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY"}

## for each year, get the number of cases for each state, and plot the number of cases for each state

## get unqiue years
years = data["Year"].unique()

## get the state names
state_names = data["State"].unique()

## add new column for the state names initials
data["State_initials"] = data["State"].apply(lambda x: state_name_dict[x.lower()])

## for each year , plot a bar chart for the number of cases for each state
for year in years:
    ## plot the bar plot with x axis is state, y axis is the number of cases
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data["State_initials"], y=data[data["Year"] == year]["Neuroinvasive_disease_cases"], name="Neuroinvasive disease cases"))
    fig.update_layout(title="The number of Neuroinvasive disease cases in each state in " + str(year), barmode="group")

    # Set y-axis limits
    fig.update_yaxes(range=[0, 1200])

    # Rotate x-axis tick labels by 45 degrees
    fig.update_xaxes(tickangle=45)

    ## save the plot as png file
    fig.write_image("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/result/basic_stats_orginal_data/Neuroinvasive_disease_" + str(year) + ".png")

## read an image sequence where where the first picture start with human_wnv_2000.png to human_wnv_2023
images = glob.glob("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/result/basic_stats_orginal_data/Neuroinvasive_disease_*.png")

## sort the images by name
images.sort()

first_image = cv2.imread(images[0])

# Obtain frame size information using get() method
frame_width = int(first_image.shape[1])
frame_height = int(first_image.shape[0])
frame_size = (frame_width, frame_height)
fps = 2

# Initialize video writer object
output = cv2.VideoWriter('/Users/ericliao/Desktop/dissertation/proposal defense/images/wnv_human_cases_report_state_over_years.avi',
                         cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, frame_size)

for image_name in images:
    img = cv2.imread(image_name)
    output.write(img)

output.release()