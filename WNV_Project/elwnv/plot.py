from matplotlib import pyplot as plt
import numpy as np

class Plot:
    def __init__(self):
        # TODO: Pyplot is a state machine, but we instantiate it in the constructor, which is not ideal.
        # prepare for plotting
        plt.style.use('seaborn-whitegrid')
        # Set Matplotlib defaults
        plt.rc('figure', autolayout=True)
        plt.rc('axes', labelweight='bold', labelsize='large',
               titleweight='bold', titlesize=18, titlepad=10)
        plt.rc('animation', html='html5')
    def show(self):

        plt.savefig("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/plots/loss_function/"
                    "cross_mse_all_counties_pca.png", dpi=300)
        plt.show()

    def sensitivity(self, train, test, model, test_column_names):
        # calculate the vaiance of the training data
        std_train = np.std(train, axis=0)

        # based on the deviation from norm layer, add 100 noise point to each sample
        noise = np.random.normal(scale=std_train * 0.1, size=(100,) + test.shape)

        # add noise to the test data
        noisy_test = test + noise

        # predict the model with noisy data
        preds_noise = model.predict(noisy_test.reshape((-1, 167))).reshape((100, test.shape[0]))

        # for each feature, calculate the least squared regression line in numpy linalg
        # and calculate the slope of the line

        # creat a numpy array with 1001 rows and columns of training data
        slopes = np.zeros((test.shape[0], test.shape[1]))
        # loop through each sample
        for i in range(test.shape[0]):
            slope = np.linalg.lstsq(noisy_test[:, i, :], preds_noise[:, i], rcond=None)[0]
            slopes[i] = slope

        # plot each slop as a bar chart and line and total 5 * 34 sub plots, each subplot has number of samples in test bars
        fig, ax = plt.subplots(5, 34, figsize=(100, 50))
        for i in range(test.shape[1]):
            # sort the slope in descending order
            sorted_slopes = np.sort(slopes[:, i])[::-1]
            # plot the bar chart
            ax[i // 34, i % 34].bar(range(test.shape[0]), sorted_slopes)
            # plot the line
            ax[i // 34, i % 34].plot(range(test.shape[0]), sorted_slopes)
            # set the title and use the column name as the title
            ax[i // 34, i % 34].set_title(test_column_names[i])
            # log scale y axis considering there are negative values
            ax[i // 34, i % 34].set_yscale("log")

        # save the figure
        fig.savefig(
            "/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/plots/"
            "sensitivity_analysis_0.1.png", dpi=300)
