import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# scatter plot the longitude and latitude for disease data
df = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/illinois_data/aggregate_by_county/all_years/mos_illinois_county_02_to_22.csv")

# for each year, plot the scatter plot, and plot all subplots in one figure, and order by year
# create a grid for subplots 4 by 5
fig, axs = plt.subplots(5, 5, figsize=(20, 20))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
unique_years = df["Year"].unique()
# order the unique_years
unique_years = np.sort(unique_years)
for i in range(len(unique_years)):
    df1 = df[(df["Year"] == unique_years[i])]
    axs[i//5, i % 5].scatter(df1["County_Seat_Longitude"], df1["County_Seat_Latitude"], s=df1["Mosquito"]*2, c=df1["Mosquito"], cmap="tab20b", alpha=0.5)
    # add color bar and make it horizontal at the bottom
    fig.colorbar(axs[i // 5, i % 5].scatter(df1["County_Seat_Longitude"], df1["County_Seat_Latitude"],
                                            s=df1["Mosquito"] * 1, c=df1["Mosquito"], cmap="tab20b", alpha=0.5),
                 ax=axs[i // 5, i % 5], orientation="horizontal", fraction=0.05, pad=0.05)

    # change the range of the color bar
    axs[i//5, i % 5].set_title("WNV Mosquitoes cases in {}".format(unique_years[i]))
    axs[i//5, i%5].set_xlabel("Longitude")
    axs[i//5, i%5].set_ylabel("Latitude")
    axs[i//5, i%5].set_xticklabels([])
    axs[i//5, i%5].set_yticklabels([])
    axs[i//5, i%5].set_xticks([])
    axs[i//5, i%5].set_yticks([])
    axs[i//5, i%5].set_xlim(-91.5, -87.5)
    axs[i//5, i%5].set_ylim(36.5, 42.5)
    axs[i//5, i%5].set_aspect("equal")
    axs[i//5, i%5].set_xticks(np.arange(-91.5, -87.5, 0.5))
    axs[i//5, i%5].set_yticks(np.arange(36.5, 42.5, 0.5))
    axs[i//5, i%5].tick_params(axis="both", which="major", labelsize=8)
    axs[i//5, i%5].tick_params(axis="both", which="minor", labelsize=8)
    axs[i//5, i%5].grid(True)


# ## remove the last row of subplots
fig.delaxes(axs[4, 4])
fig.delaxes(axs[4, 3])
fig.delaxes(axs[4, 2])
fig.delaxes(axs[4, 1])
fig.delaxes(axs[4, 0])


# ##save the figure
plt.savefig("/Users/ericliao/Desktop/WNV_project_files/illinois_data/plots_for_illinois_data/mos/"
            "mos_wnv_all_years.png",
            dpi=1200,
            bbox_inches="tight")


## plot the scatter plot for each year
unique_years = df["Year"].unique()
for year in unique_years:
    df1 = df[(df["Year"] == year)]
    plt.scatter(df1["County_Seat_Longitude"], df1["County_Seat_Latitude"],
                s=df1["Mosquito"]*10,
                c=df1["Mosquito"],
                cmap="tab20b",
                alpha=0.5)
    # add color bar
    plt.colorbar()
    # change the range of the color bar
    plt.clim(0, 10)
    plt.title("WNV cases for mosquitoes in {}".format(year))
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig("/Users/ericliao/Desktop/WNV_project_files/illinois_data/plots_for_illinois_data/mos/illinois_mos_in_year_{}.png".format(year), dpi=300)
    plt.close()
#######################################################################################################################
