import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_addons import metrics
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
                   "add_0_for_no_wnv/cali_week_wnnd_multi_years_all_features_ebirds.csv", index_col=0)

# drop columns that are not features and drop target
data = data.drop(["State", "County", "Year", 'Month', "County_Seat_Latitude", "County_Seat_Longitude", "FIPS",
                  # "Human_WNND_Count",
                  "Human_WNND_Rate"
                  # "Population"
                  ], axis=1)

# drop the columns that name contains "4m_shift"
# data = data.drop([col for col in data.columns if "4m_shift" in col], axis=1)

data = data.dropna()

# convert "Date" column to datetime
data["Date"] = pd.to_datetime(data["Date"])

# check the standard deviation of the target
print("The standard deviation of the target is: ", data["Human_WNND_Count"].std())

## get the "Date" before 2012-05-01 as train data
train = data[(data["Date"] > "2004-01-01") & (data["Date"] < "2012-01-01")]
test = data[data["Date"] >= "2012-01-01"]


# Get labels
train_labels = train.pop("Human_WNND_Count").values
test_labels = test.pop("Human_WNND_Count").values

# test_populations = test.pop("Population")
# train_populations = train.pop("Population")
train.pop("Date")
test.pop("Date")

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
pipe = layers.Dense(16, activation='selu')(pipe)

# pipe = layers.Dense(32)(pipe)
# pipe = layers.LeakyReLU()(pipe)
pipe = layers.BatchNormalization()(pipe)
pipe = layers.GaussianDropout(0.3)(pipe)
#
# pipe = layers.Dense(32)(pipe)
# pipe = layers.LeakyReLU()(pipe)
pipe = layers.Dense(16, activation='selu')(pipe)
pipe = layers.BatchNormalization()(pipe)
pipe = layers.GaussianDropout(0.3)(pipe)

# pipe = layers.Dense(12, activation="relu")(pipe)
# pipe = layers.BatchNormalization()(pipe)
# pipe = layers.GaussianDropout(0.1)(pipe)

## use exponential activation function for regression to shrink big numbers of target "COUNT"
pipe = layers.Dense(1, activation='selu')(pipe)

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

df_out = pd.DataFrame(dict(
    pred_WNV=model.predict(test)[:, -1],
    actual_WNV=test_labels,
    # population=test['Population']
))

# # add two columns equals to population * pred_WNV and population * actual_WNV
# df_out["pred_WNV_count"] = df_out["pred_WNV"] * df_out["population"]
# df_out["actual_WNV_count"] = df_out["actual_WNV"] * df_out["population"]

df_out.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/add_0_for_no_wnv/"
          "nn_cali_multi_Years_result.csv", index=False)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")

plt.savefig("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
            "add_0_for_no_wnv/cross_mae_multi_Years.png", dpi=300)

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







