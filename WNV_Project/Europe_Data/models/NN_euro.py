## build a NN model for Europe data
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/Europe_data/adding_0_case/"
                   "europe_data_with_coordinates_landuse_climate_0_case.csv", index_col=0)

# select the columns as predictors
data_pred = data.iloc[:, 11:]

# convert "Time" column contains year value to datetime
data_pred["Time"] = pd.to_datetime(data["Time"], format="%Y")
data_pred["NumValue"] = data["NumValue"]

# drop nan values
data_pred = data_pred.dropna()

### train and test data ####
train = data_pred[(data_pred["Time"] < "2020")]
test = data_pred[(data_pred["Time"] >= "2020")]

# Get labels
train_labels = train.pop("NumValue").values
test_labels = test.pop("NumValue").values

# convert both train and test labels to float32
train_labels = tf.cast(train_labels, tf.float32)
test_labels = tf.cast(test_labels, tf.float32)

## remove time column
train.pop("Time")
test.pop("Time")

# get the column names
train_column_names = train.columns
test_column_names = test.columns

# get the shape of train and test for neural network input layer
train_column_number = train.shape[1]
test_column_number = test.shape[1]

# given epslon to avoid divide by zero
epslon = 1e-6

## sample weights
def custom_loss(y_true, y_pred):
    threshold = 5
    return tf.reduce_mean(tf.where(tf.greater(y_true, threshold),
                                   tf.square(y_true - y_pred),
                                   tf.sqrt(tf.abs(y_true - y_pred))))

## prepare for plottting
plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

## Define the input layers for non-categorical features and add batch normalization
input = pipe = layers.Input(shape=train_column_number, name="noncat")

## normalize the features
pipe = tf.keras.layers.Normalization()(pipe)

## Create the hidden layer the rest of the neural network layers
pipe = layers.Dense(64, activation='selu')(pipe)
pipe = layers.BatchNormalization()(pipe)
# pipe = layers.GaussianDropout(0.3)(pipe)
#
pipe = layers.Dense(64, activation='selu')(pipe)
pipe = layers.BatchNormalization()(pipe)
# pipe = layers.GaussianDropout(0.3)(pipe)

## use exponential activation function for regression to shrink big numbers of target "COUNT"
pipe = layers.Dense(1, activation='selu')(pipe)

## build model
model = keras.models.Model(inputs=input, outputs=[pipe])

## compile model using regression metrics
model.compile(
    optimizer="adam",
    loss=custom_loss,
    metrics=["mse", "mae", "mape"],
    # weighted_metrics=["mse", "mae", "mape"],
)

## fit the model
history = model.fit(
    train,
    train_labels,
    batch_size=128,
    epochs=200,
    verbose=1,
    validation_split=0.2,
    # sample_weight=weight,
)

## plot the loss and mse
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
# log the y axis
plt.yscale('log')
plt.legend()
plt.title("Loss")
plt.show()

## plot the mse
plt.plot(history.history['mse'], label='train')
plt.plot(history.history['val_mse'], label='validation')
plt.legend()
plt.title("MSE")
plt.show()

## plot the mae
plt.plot(history.history['mae'], label='train')
plt.plot(history.history['val_mae'], label='validation')
plt.legend()
plt.title("MAE")
plt.show()

## plot the mape
plt.plot(history.history['mape'], label='train')
plt.plot(history.history['val_mape'], label='validation')
plt.legend()
plt.title("MAPE")
plt.show()

## get the predicted values
y_predict = model.predict(test)

## store test labels and predicted labels in a dataframe
df = pd.DataFrame({"test_labels": test_labels, "y_predict": y_predict[:, 0]})
df.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/Europe_data/adding_0_case/neural_network_0_case/results/CNN_euro_result{}.csv".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))



