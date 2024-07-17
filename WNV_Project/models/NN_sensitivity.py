import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_addons import metrics
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/Data/"
                   "human_neuroinvasive_wnv_ebirds.csv", index_col=0)

### southern_california_counties = ["Los Angeles", "San Diego", "Orange", "Riverside", "San Bernardino", "Kern", "Ventura","Santa Barbara", "San Luis Obispo", "Imperial"]

### in FIPS
southern_california_counties = [6037, 6073, 6059, 6065, 6071, 6029, 6111, 6083, 6079, 6025]

# data = data[data["FIPS"].isin(southern_california_counties) | data["State"].isin(['North Dakota', 'South Dakota', 'Colorado'])]

data = data[data["State"] == "California"]

# # drop columns that are not features and drop target
data = data.drop(["FIPS", "County", "State", "State_Code", 'SET', "County_Seat", "County_Seat_Latitude",
                  "County_Seat_Longitude", "County_Centroid_Latitude", "County_Centroid_Longitude",
                  'Poverty_Estimate_All_Ages',
                  # "Population",
                  "State_Land_Area", "Land_Change_Count_Since_1992",
                  "Land_Use_Class", "Processed_Flag_Land_Use", "WNV_Rate_Neural_With_All_Years",
                  # "WNV_Rate_Neural_Without_99_21",
                  "WNV_Rate_Non_Neural_Without_99_21",
                  "State_Horse_WNV_Rate", "WNV_Rate_Non_Neural_Without_99_21_log",
                  "WNV_Rate_Neural_Without_99_21_log"# target column
                  ], axis=1)


### drop monthly weather data block #######################
## get the column u10_Jan and column swvl1_Dec index
column_Poverty_index = data.columns.get_loc("Poverty_Rate_Estimate_All_Ages")
column_u10_Jan_index = data.columns.get_loc("u10_Jan")
column_swvl1_Dec_index = data.columns.get_loc("swvl1_Dec")
column_tp_acc_extrem_index = data.columns.get_loc("tp_acc_Oct_to_Aug")

## DROP the columns between column_u10_Jan and column_swvl1_Dec includes column_u10_Jan and column_swvl1_Dec
# data = data.drop(data.columns[column_u10_Jan_index:column_swvl1_Dec_index + 1], axis=1)
data = data.drop(data.columns[column_u10_Jan_index:column_swvl1_Dec_index + 1], axis=1)
################################################################

data = data.dropna()

train = data[(data["Year"] < 2020) & (data["Year"] >= 2003)]
test = data[(data["Year"] >= 2020) & (data["Year"] < 2022)]

# Get labels
train_labels = train.pop("WNV_Rate_Neural_Without_99_21").values
test_labels = test.pop("WNV_Rate_Neural_Without_99_21").values

train.pop("Year")
test.pop("Year")

train_population = train.pop("Population").values
test_population = test.pop("Population").values

# get the column names
train_column_names = train.columns
test_column_names = test.columns

# get the shape of train and test for neural network input layer
train_column_number = train.shape[1]
test_column_number = test.shape[1]

# given epslon to avoid divide by zero
epslon = 1e-6

## prepare for plottting
plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

## prepare for neural network
input_non_categorical_shape = (train.shape[1],)

## Define the input layers for non-categorical features and add batch normalization
noncat_input = pipe = layers.Input(shape=input_non_categorical_shape, name="noncat")
pipe = layers.BatchNormalization()(pipe)

## Create the hidden layer the rest of the neural network layers
pipe = layers.Dense(10, activation='relu')(pipe)
pipe = layers.BatchNormalization()(pipe)
pipe = layers.GaussianDropout(0.1)(pipe)

# pipe = layers.Dropout(0.4)(pipe)
# pipe = layers.Dense(8, activation="relu")(pipe)
# pipe = layers.BatchNormalization()(pipe)
# pipe = layers.GaussianDropout(0.1)(pipe)
# pipe = layers.Dropout(0.4)(pipe)

## use exponential activation function for regression to shrink big numbers of target "COUNT"
pipe = layers.Dense(1, activation="relu")(pipe)

## build model
model = keras.models.Model(inputs=noncat_input, outputs=[pipe])

## compile model using regression metrics
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mse", "mae", "mape"],
)

# early stopping
early_stopping = keras.callbacks.EarlyStopping(
    patience=20,
    min_delta=0.001,
    restore_best_weights=True,
)

# fit model
history = model.fit(
    train,
    train_labels,
    validation_data=(test, test_labels),
    batch_size=32,
    epochs=1000,
    callbacks=[early_stopping]
)

pd.DataFrame(dict(
    pred_WNV=model.predict(test)[:, -1],
    actual_WNV=test_labels,
    population=test_population,
)).to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results"
          "/prediction_results_all_counties.csv", index=False)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")

plt.savefig("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/plots/loss_function/"
            "cross_mae_all_counties_all_counties.png", dpi=300)

plt.show()
######################################################################
#
# ###### sensitivity analysis ########
# # calculate the vaiance of the training data
# std_train = np.std(train, axis=0)
#
# #based on the deviation from norm layer, add 100 noise point to each sample
# noise = np.random.normal(scale=std_train * 0.1, size=(100,) + validation.shape)
#
# # add noise to the test data
# noisy_validation = validation + noise
#
# # predict the model with noisy data
# preds_noise = model.predict(noisy_validation.reshape((-1, input_non_categorical_shape[0]))).reshape((100, validation.shape[0]))
#
# # for each feature, calculate the least squared regression line in numpy linalg
# # and calculate the slope of the line
#
# # creat a numpy array with 1001 rows and columns of training data
# slopes = np.zeros((validation.shape[0], validation.shape[1]))
# # loop through each sample
# for i in range(validation.shape[0]):
#     slope = np.linalg.lstsq(noisy_validation[:, i, :], preds_noise[:, i], rcond=None)[0]
#     slopes[i] = slope
#
# # plot each slop as a bar chart and line and total 4 * 4 sub plots, each subplot has number of samples in test bars
# fig, ax = plt.subplots(4, 4, figsize=(20, 20))
# for i in range(test.shape[1]):
#     # sort the slope in descending order
#     sorted_slopes = np.sort(slopes[:, i])[::-1]
#     # plot the bar chart
#     ax[i // 4, i % 4].bar(range(validation.shape[0]), sorted_slopes)
#     # plot the line chart
#     ax[i // 4, i % 4].plot(range(validation.shape[0]), sorted_slopes)
#     # set the title
#     ax[i // 4, i % 4].set_title(validation_column_names[i])
#     # set y axis in log scale
#     ax[i // 4, i % 4].set_yscale("symlog")
#     # set y axis limit based on the max and min of the slope,
#     # if the slope is inf or nan, do not set y lim
#     if np.isfinite(sorted_slopes[0]) | np.isfinite(sorted_slopes[-1]):
#         continue
#     else:
#         ax[i // 4, i % 4].set_ylim(sorted_slopes[0], sorted_slopes[-1])
#
#
#
# # save the figure
# fig.savefig("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_13_to_2023_02_20/"
#             "results/sensitivity_analysis_gaussion_0.4_dropout_validation_early_stop.png", dpi=300)







