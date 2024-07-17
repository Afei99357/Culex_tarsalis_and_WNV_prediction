import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_addons import metrics
import tensorflow as tf
import matplotlib.pyplot as plt

## prepare for plottting
plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

# Load dataset
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_06/"
                   "human_neuroinvasive_and_horse_0_case_whole_year.csv", index_col=0)

# drop rows contains nan
data = data.dropna()

# # only on horse
human_data = data[
    (data["SET"] == "VET")
    & (data["Year"] < 2019)
    ]

# # find Counties in southern california
# southern_california_counties = ["Los Angeles", "San Diego", "Orange", "Riverside", "San Bernardino", "Kern", "Ventura",
#                                 "Santa Barbara", "San Luis Obispo", "Imperial"]
#
# ### only southern california data
# human_data = human_data[human_data["County"].isin(southern_california_counties) | train["State"].isin(['North Dakota', 'South Dakota'])]

human_data = human_data.reset_index(drop=True)

# Get count labels
data_labels = human_data.WNV_Count.values
## get categorical feature 'land use class' and "land change" and keep the original index
land_use_class_data = human_data.pop("Land_Use_Class").to_numpy()
land_change_data = human_data.pop("Land_Change_Count_Since_1992").to_numpy()

# get unique years in human data and order them as ascending
unique_years = np.unique(human_data["Year"])
unique_years.sort()

# get the column Year
year_column = human_data.pop("Year").to_frame()

# drop columns
human_data = human_data.drop(["FIPS", "County", "State", "State_Code", "WNV_Count", "County_Centroid_Latitude",
                    "County_Centroid_Longitude", "Processed_Flag_Land_Use", 'SET'
                              , "Poverty_Estimate_All_Ages"], axis=1)

# TODO : fit model on one year and predict the next year, then use the next year as training data and predict the next year
# TODO : repeat this process until the last year
## remove the comma in the population column and Poverty_Estimate_All_Ages column and convert to float
human_data["Population"] = human_data["Population"].str.replace(",", "")
human_data["State_Land_Area"] = human_data["State_Land_Area"].str.replace(",", "")
# ### for human data
# human_data["Poverty_Estimate_All_Ages"] = human_data["Poverty_Estimate_All_Ages"].str.replace(",", "")

## convert population and Povert_Estimate_All_Ages column to float
human_data["Population"] = human_data["Population"].astype(float)
human_data["State_Land_Area"] = human_data["State_Land_Area"].astype(float)
# ### for human data
# human_data["Poverty_Estimate_All_Ages"] = human_data["Poverty_Estimate_All_Ages"].astype(float)


human_data = human_data.to_numpy()

# get unique land use class and land change
unique_categorical_land_use = np.unique(land_use_class_data)
unique_categorical_land_change = np.unique(land_change_data)

## prepare for neural network
input_non_categorical_shape = (human_data.shape[1],)

# input_land_use_train = np.searchsorted(unique_categorical_land_use, land_use_class_train)

## Define the input layers for land use class
cat_land_use_class_input = pipe1 = layers.Input(shape=(1,), name="cat_land_use_class")
## embedding layer for land use class
pipe1 = layers.Embedding(unique_categorical_land_use.size, 256, name='land_use_embedding')(pipe1)
pipe1 = tf.reshape(pipe1, [-1, 256])

## Define the input layers for land change
cat_land_change_input = pipe2 = layers.Input(shape=(1,), name="cat_land_change")
## embedding layer for land change
pipe2 = layers.Embedding(unique_categorical_land_change.size, 256, name="land_change_embedding")(pipe2)
pipe2 = tf.reshape(pipe2, [-1, 256])

## Define the input layers for non-categorical features and add batch normalization
noncat_input = pipe = layers.Input(shape=input_non_categorical_shape, name="noncat")
pipe = layers.BatchNormalization()(pipe)
## Create the hidden layer the rest of the neural network layers
pipe = layers.Dense(256, activation='relu')(pipe)
pipe = pipe + pipe1 + pipe2
pipe = layers.BatchNormalization()(pipe)
pipe = layers.Dense(128, activation="relu")(pipe)
pipe = layers.BatchNormalization()(pipe)
pipe = layers.Dropout(0.2)(pipe)
pipe = layers.Dense(64, activation="relu")(pipe)
pipe = layers.BatchNormalization()(pipe)
pipe = layers.Dropout(0.2)(pipe)
pipe = layers.Dense(32, activation="relu")(pipe)
pipe = layers.BatchNormalization()(pipe)
pipe = layers.Dropout(0.2)(pipe)

## use exponential activation function for regression to shrink big numbers of target "COUNT"
pipe = layers.Dense(1, activation=tf.exp)(pipe)
# pipe = layers.Dense(1)(pipe)
## build model
model = keras.models.Model(inputs=[cat_land_use_class_input, cat_land_change_input, noncat_input], outputs=[pipe])

# compile model
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mse", "mae", "mape"],
)

# early stopping
early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)

# create a function to  get the train and test data index for each year
def get_train_test_index(data, current_year):
    # get the index in the data that has the current year
    train_index = data[data["Year"] == current_year].index.values
    test_index = data[data["Year"] == current_year + 1].index.values
    return train_index, test_index

# get train and test index
for year in unique_years[:-1]:
    train_index, test_index = get_train_test_index(year_column, year)

    # build the dictionary for train and test data
    land_use_class_train = land_change_data[train_index]
    land_use_class_test = land_change_data[test_index]
    # use search sorted to convert input_categorical_land_use_shape to continuous number
    input_land_use_train = np.searchsorted(unique_categorical_land_use, land_use_class_train)
    input_land_use_test = np.searchsorted(unique_categorical_land_use, land_use_class_test)

    input_train_dict = {"cat_land_use_class": input_land_use_train, "cat_land_change": land_change_data[train_index],
                        "noncat": human_data[train_index]}
    input_test_dict = {"cat_land_use_class": input_land_use_test, "cat_land_change": land_change_data[test_index],
                        "noncat": human_data[test_index]}

    train_labels = data_labels[train_index]
    test_labels = data_labels[test_index]

    # fit model
    history = model.fit(
        input_train_dict,
        train_labels,
        validation_data=(input_test_dict, test_labels),
        batch_size=64,
        epochs=1000,
        callbacks=[early_stopping]
    )

    pd.DataFrame(dict(
        pred_mosq=model.predict(input_test_dict)[:, -1],
        actual_mosq=test_labels,
    )).to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_06/horse_only_results/mse/NN_wnv_horse_{}_mse.csv".format(year),
              index=False)

    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
    # add title
    plt.title("Cross-entropy_model_{}".format(year))

    # save figure
    plt.savefig("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_06/horse_only_results/"
                "mse/NN_wnv_horse_{}_mse.png".format(year))

    plt.show()


