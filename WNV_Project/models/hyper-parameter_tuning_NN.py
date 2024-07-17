import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow_addons import metrics
import tensorflow as tf
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp
from tensorflow.keras.models import Sequential


# Load dataset
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_13_to_2023_02_20/"
                   "dataset/human_neuroinvasive_with_extreme_weather_with_county_seat_modify.csv", index_col=False)

# drop rows contains nan
data = data.dropna()

southern_california_counties = ["Los Angeles", "San Diego", "Orange", "Riverside", "San Bernardino", "Kern", "Ventura",
                                        "Santa Barbara", "San Luis Obispo", "Imperial"]

data = data[data["County"].isin(southern_california_counties) | data["State"].isin(['North Dakota', 'South Dakota', 'Colorado'])]


# split into train and test
train = data[(data["Year"] < 2017) & (data["Year"] >= 2003)]
test = data[(data["Year"] >= 2017) & (data["Year"] < 2022)]

# Get labels
train_labels = train.WNV_Count.values
test_labels = test.WNV_Count.values

# drop columns
drop_list = ["FIPS", "County", "State", "State_Code", "Year", "WNV_Count",
                          # "Non_Neural_WNV_COUNT", "Neural_WNV_Count",
                          "County_Centroid_Latitude", "County_Centroid_Longitude", "County_Seat_Latitude",
                          "County_Seat_Longitude", "County_Seat", "Processed_Flag_Land_Use", 'SET',
                          'Poverty_Estimate_All_Ages']

train = train.drop(drop_list, axis=1)
test = test.drop(drop_list, axis=1)

# get categorical feature 'land use class' and "land change"
land_change_train = train.pop("Land_Change_Count_Since_1992").to_numpy()
land_change_test = test.pop("Land_Change_Count_Since_1992").to_numpy()

land_use_class_train = train.pop("Land_Use_Class").to_numpy()
land_use_class_test = test.pop("Land_Use_Class").to_numpy()

# get the column names
train_column_names = train.columns
test_column_names = test.columns

# get the shape of train and test for neural network input layer
train_column_number = train.shape[1]
test_column_number = test.shape[1]

train = train.to_numpy()
test = test.to_numpy()

# Calculate the number of unique values in the categorical features for embedding input layer
unique_categorical_land_use = np.unique(np.concatenate((land_use_class_train, land_use_class_test), axis=0))
# use search sorted to convert input_categorical_land_use_shape to continuous number
input_land_use_train = np.searchsorted(unique_categorical_land_use, land_use_class_train)
input_land_use_test = np.searchsorted(unique_categorical_land_use, land_use_class_test)

unique_categorical_land_change_shape = np.unique(np.concatenate((land_change_train, land_change_test), axis=0))

# Define the search space
space = {
    'num_layers': hp.quniform('num_layers', 2, 4, 1),
    'num_neurons': hp.quniform('num_neurons', 8, 128, 8),
    'dropout': hp.uniform('dropout', 0, 0.5),
}

# create a dataframee to store the result
result = pd.DataFrame(columns=['num_layers', 'num_neurons', 'dropout', 'loss', 'val_loss', 'mae', 'val_mae'])


# Define the objective function
def objective(params):
    num_layers = int(params['num_layers'])
    num_neurons = int(params['num_neurons'])
    dropout = params['dropout']

    #print out all three parameters above
    print('num_layers:', num_layers)
    print('num_neurons:', num_neurons)
    print('dropout:', dropout)

    ## prepare for neural network
    input_non_categorical_shape = (train.shape[1],)

    # Define the input layers for land use class
    land_use = layers.Input(shape=(1,), name="cat_land_use_class")

    # Define the input layers for land change
    land_change = layers.Input(shape=(1,), name="cat_land_change")

    # Define the input layers for non-categorical features and add batch normalization
    noncat = layers.Input(shape=input_non_categorical_shape, name="noncat")

    # embedding layer for Species
    pipe1 = layers.Embedding(unique_categorical_land_use.size, num_neurons)(land_use)
    pipe1 = tf.reshape(pipe1, [-1, num_neurons])

    # embedding layer for land change
    pipe2 = layers.Embedding(unique_categorical_land_change_shape.size, num_neurons)(land_change)
    pipe2 = tf.reshape(pipe2, [-1, num_neurons])

    pipe = layers.BatchNormalization()(noncat)
    # Create the hidden layer the rest of the neural network layers
    pipe = layers.Dense(num_neurons, activation='relu')(pipe)

    # adds all the input
    pipe = pipe + pipe1 + pipe2
    # Define the model architecture
    nn_model = Sequential()
    # add batch normalization
    nn_model.add(layers.BatchNormalization())
    for i in range(num_layers):
        nn_model.add(layers.Dense(num_neurons, activation='relu'))
        nn_model.add(layers.GaussianDropout(dropout))

    pipe = nn_model(pipe)
    pipe = layers.Dense(1, activation=tf.exp)(pipe)

    model = models.Model(inputs=[land_use, land_change, noncat], outputs=[pipe])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='msle',
                  metrics=["mse"])

    # early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        patience=10,
        min_delta=0.001,
        restore_best_weights=True,
    )

    input_train_dict = {"cat_land_use_class": input_land_use_train, "cat_land_change": land_change_train,
                        "noncat": train}
    input_test_dict = {"cat_land_use_class": input_land_use_test, "cat_land_change": land_change_test,
                       "noncat": test}

    # Train the model
    history = model.fit(
        input_train_dict,
        train_labels,
        validation_data=(input_test_dict, test_labels),
        batch_size=32,
        epochs=100,
        # callbacks=[early_stopping]
    )

    # Evaluate the model on the validation set
    val_loss, val_mse = model.evaluate(input_test_dict, test_labels)

    # save the result
    result.loc[len(result)] = [num_layers, num_neurons, dropout, history.history['loss'][-1], history.history['val_loss'][-1], history.history['mse'][-1], history.history['val_mse'][-1]]

    # output the result to csv
    result.to_csv('/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_13_to_2023_02_20/results/result.csv', index=False)

    return val_loss

# Run the hyperparameter search using TPE algorithm
best = fmin(objective, space, algo=tpe.suggest, max_evals=50)

print('Best hyperparameters:', best)






