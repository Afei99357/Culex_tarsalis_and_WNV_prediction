import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_addons import metrics
import tensorflow as tf
import matplotlib.pyplot as plt


# Load dataset
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_02/"
                   "human_neuroinvasive_and_horse_0_case_whole_year.csv", index_col=0)

# drop rows contains nan
data = data.dropna()

# split into train and test
train = data[data["Year"] < 2017]
test = data[(data["Year"] >= 2017) & (data["Year"] < 2018)]

# find Counties in southern california
southern_california_counties = ["Los Angeles", "San Diego", "Orange", "Riverside", "San Bernardino", "Kern", "Ventura",
                                "Santa Barbara", "San Luis Obispo", "Imperial"]

train = train[train["County"].isin(southern_california_counties) | train["State"].isin(['North Dakota', 'South Dakota'])]
test = test[test["County"].isin(southern_california_counties) | test["State"].isin(['North Dakota', 'South Dakota'])]

# # only on horse
# train = train[train["SET"] == "VET"]
# test = test[test["SET"] == "VET"]

# only on human
train = train[train["SET"] == "HUMAN"]
test = test[test["SET"] == "HUMAN"]

# Get labels
train_labels = train.WNV_Count.values
test_labels = test.WNV_Count.values

# drop columns
train = train.drop(["FIPS", "County", "State", "State_Code", "Year", "WNV_Count", "County_Centroid_Latitude",
                    "County_Centroid_Longitude", "Processed_Flag_Land_Use", 'SET', 'Poverty_Estimate_All_Ages'], axis=1)
test = test.drop(["FIPS", "County", "State", "State_Code", "Year", "WNV_Count", "County_Centroid_Latitude",
                  "County_Centroid_Longitude", "Processed_Flag_Land_Use", "SET", 'Poverty_Estimate_All_Ages'], axis=1)

# get the column names
train_column_names = train.columns
test_column_names = test.columns

# get the shape of train and test for neural network input layer
train_column_number = train.shape[1]
test_column_number = test.shape[1]

## todo: decide if Land_Change_Count_Since_1992 shuold concider as categorical or not?
land_change_train = train.pop("Land_Change_Count_Since_1992").to_numpy()
land_change_test = test.pop("Land_Change_Count_Since_1992").to_numpy()

## get categorical feature 'land use class' and "land change"
land_use_class_train = train.pop("Land_Use_Class").to_numpy()
land_use_class_test = test.pop("Land_Use_Class").to_numpy()

## remove the comma in the population column and Poverty_Estimate_All_Ages column and convert to float
train["Population"] = train["Population"].str.replace(",", "")
test["Population"] = test["Population"].str.replace(",", "")

### for human data
# train["Poverty_Estimate_All_Ages"] = train["Poverty_Estimate_All_Ages"].str.replace(",", "")
# test["Poverty_Estimate_All_Ages"] = test["Poverty_Estimate_All_Ages"].str.replace(",", "")

train["State_Land_Area"] = train["State_Land_Area"].str.replace(",", "")
test["State_Land_Area"] = test["State_Land_Area"].str.replace(",", "")

## convert population and Povert_Estimate_All_Ages column to float
train["Population"] = train["Population"].astype(float)
test["Population"] = test["Population"].astype(float)

### for human data
# train["Poverty_Estimate_All_Ages"] = train["Poverty_Estimate_All_Ages"].astype(float)
# test["Poverty_Estimate_All_Ages"] = test["Poverty_Estimate_All_Ages"].astype(float)

train["State_Land_Area"] = train["State_Land_Area"].astype(float)
test["State_Land_Area"] = test["State_Land_Area"].astype(float)

train = train.to_numpy()
test = test.to_numpy()

## prepare for plottting
plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

## prepare for neural network
input_non_categorical_shape = (train.shape[1],)

## prepare for neural network
input_non_categorical_shape = (train.shape[1],)

## Calculate the number of unique values in the categorical features for embedding input layer
unique_categorical_land_use = np.unique(np.concatenate((land_use_class_train, land_use_class_test), axis=0))
# use search sorted to convert input_categorical_land_use_shape to continuous number
input_land_use_train = np.searchsorted(unique_categorical_land_use, land_use_class_train)
input_land_use_test = np.searchsorted(unique_categorical_land_use, land_use_class_test)


unique_categorical_land_change_shape = np.unique(np.concatenate((land_change_train, land_change_test), axis=0))

## Define the input layers for land use class
cat_land_use_class_input = pipe1 = layers.Input(shape=(1,), name="cat_land_use_class")
## embedding layer for Species
pipe1 = layers.Embedding(unique_categorical_land_use.size, 256, name='land_use_embedding')(pipe1)
pipe1 = tf.reshape(pipe1, [-1, 256])

## Define the input layers for land change
cat_land_change_input = pipe2 = layers.Input(shape=(1,), name="cat_land_change")
## embedding layer for land change
pipe2 = layers.Embedding(unique_categorical_land_change_shape.size, 256, name="land_change_embedding")(pipe2)
pipe2 = tf.reshape(pipe2, [-1, 256])

## Define the input layers for non-categorical features and add batch normalization
noncat_input = pipe = layers.Input(shape=input_non_categorical_shape, name="noncat")
pipe = layers.BatchNormalization()(pipe)
## Create the hidden layer the rest of the neural network layers
pipe = layers.Dense(256, activation='relu')(pipe)
pipe = pipe + pipe1 + pipe2
pipe = layers.BatchNormalization()(pipe)
pipe = layers.Dense(256, activation="relu")(pipe)
pipe = layers.BatchNormalization()(pipe)
pipe = layers.Dropout(0.2)(pipe)
pipe = layers.Dense(256, activation="relu")(pipe)
pipe = layers.BatchNormalization()(pipe)
pipe = layers.Dropout(0.2)(pipe)

## use exponential activation function for regression to shrink big numbers of target "COUNT"
pipe = layers.Dense(1, activation=tf.exp)(pipe)
# pipe = layers.Dense(1)(pipe)
## build model
model = keras.models.Model(inputs=[cat_land_use_class_input, cat_land_change_input, noncat_input], outputs=[pipe])

## compile model using regression metrics
## using 2017 horse data as prediction compare to 2018 as real horse data
## the RMSE is 0.932503590234562 and STD is 1.0521133906210807
##
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

# first adapt normalization layer for preprocessing data
# norm.adapt(train)

input_train_dict = {"cat_land_use_class": input_land_use_train, "cat_land_change": land_change_train, "noncat": train}
input_test_dict = {"cat_land_use_class": input_land_use_test, "cat_land_change": land_change_test, "noncat": test}


# fit model
history = model.fit(
    input_train_dict,
    train_labels,
    validation_data=(input_test_dict, test_labels),
    batch_size=256,
    epochs=100,
    # callbacks=[early_stopping]
)

pd.DataFrame(dict(
    pred_mosq=model.predict(input_test_dict)[:, -1],
    actual_mosq=test_labels,
)).to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_02/results/NN_wnv_horse.csv", index=False)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")

plt.show()
######################################################################






