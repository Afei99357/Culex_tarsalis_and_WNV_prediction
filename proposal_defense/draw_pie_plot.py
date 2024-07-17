import matplotlib.pyplot as plt


def draw_pie_plot(ratio, labels, title):
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(ratio, labels=[''] * len(labels), autopct='%1.1f%%', startangle=90)

    # Add legend at the left upper corner
    ax.legend(wedges, labels, loc='upper left', bbox_to_anchor=(-0.1, 1), frameon=False, handletextpad=0.5,
              columnspacing=1.0)

    # Hide the autopct text
    for autotext in autotexts:
        autotext.set_color('white')

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Remove the axis
    ax.axis('off')

    plt.title(title)
    plt.show()

    # Save the pie plot
    fig.savefig('/Users/ericliao/Desktop/dissertation/proposal defense/images/plots/amoava_pie_plot.png', dpi=300,
                bbox_inches='tight')


# Example
ratio = [78.86, 2.54, 18.60]
labels = ['Within Samples', 'Between Samples within Regions', 'Between Regions']
title = 'Analysis of Molecular Variance (AMOVA) Results'

draw_pie_plot(ratio, labels, title)
