## build a NN model for Europe data
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
import seaborn as sns
import keras.backend as K

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/"
                   "human/cdc_human_1999_to_2023/WNV_human_and_non_human_yearly_climate_demographic_bird.csv", index_col=0)

## remove any space and comma in Population column
data["Population"] = data["Population"].str.replace(",", "").str.strip()

## convert Population column to numeric
data["Population"] = pd.to_numeric(data["Population"], errors='coerce')

# select the columns after column Date as predictors
date_index = data.columns.get_loc("Date")

# Select columns after the "Date" column as predictors
data_pred = data.iloc[:, date_index+1:]

# get Year column from data and add it to data_pred
data_pred['Year'] = data['Year']

## add target column
data_pred["Neuroinvasive_disease_cases"] = data["Neuroinvasive_disease_cases"]

# drop nan values
data_pred = data_pred.dropna()

## reset index
data_pred = data_pred.reset_index(drop=True)

### train and test data ####
train = data_pred[(data_pred["Year"] < 2018)]
test = data_pred[(data_pred["Year"] >= 2018)]

# Get labels
train_labels = train.pop("Neuroinvasive_disease_cases").values
test_labels = test.pop("Neuroinvasive_disease_cases").values

###convert both train and test labels to float32
train_labels = tf.constant(train_labels)
test_labels = tf.constant(test_labels)

## remove time column
train.pop("Year")
test.pop("Year")

# get the column names
train_column_names = train.columns
test_column_names = test.columns

# get the shape of train and test for neural network input layer
train_column_number = train.shape[1]
test_column_number = test.shape[1]

# given epslon to avoid divide by zero
epslon = 1e-6

# ## sample weights
# def custom_loss(y_true, y_pred):
#     threshold = 5
#     return tf.reduce_mean(tf.where(tf.greater(y_true, threshold),
#                                    tf.square(y_true - y_pred),
#                                    tf.sqrt(tf.abs(y_true - y_pred))))

def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())

## prepare for plottting
sns.set_style('whitegrid')
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
pipe = layers.Dense(32, activation='selu')(pipe)
pipe = layers.BatchNormalization()(pipe)
# pipe = layers.GaussianDropout(0.3)(pipe)

## use exponential activation function for regression to shrink big numbers of target "COUNT"
pipe = layers.Dense(1, activation='selu')(pipe)

## build model
model = keras.models.Model(inputs=input, outputs=[pipe])

## compile model using regression metrics
model.compile(
    optimizer="adam",
    loss="msle",
    metrics=["mse", "mae", "mape", r_squared],
    # weighted_metrics=["mse", "mae", "mape"],
)

## fit the model
history = model.fit(
    train,
    train_labels,
    batch_size=64,
    epochs=100,
    verbose=1,
    validation_split=0.2,
    # sample_weight=weight,
)

## plot the loss and mse
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
## get the loss function name
loss_name = model.loss
# log the y axis
plt.grid(False)
plt.yscale('log')
plt.legend()
plt.title("Loss")
plt.savefig("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/result/NN/nn_yearly_loss_"+ loss_name +".png", dpi=300)
plt.show()

## plot the mse
plt.plot(history.history['mse'], label='train')
plt.plot(history.history['val_mse'], label='validation')
plt.grid(False)
plt.legend()
plt.title("MSE")
plt.savefig("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/result/NN/nn_yearly_mse.png", dpi=300)
plt.show()

## plot the mae
plt.plot(history.history['mae'], label='train')
plt.plot(history.history['val_mae'], label='validation')
plt.grid(False)
plt.legend()
plt.title("MAE")
plt.savefig("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/result/NN/nn_yearly_mae.png", dpi=300)
plt.show()

## plot the mape
plt.plot(history.history['mape'], label='train')
plt.plot(history.history['val_mape'], label='validation')
plt.grid(False)
plt.legend()
plt.title("MAPE")
plt.savefig("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/result/NN/nn_yearly_mape.png", dpi=300)
plt.show()

# Plot the R2
plt.plot(history.history['r_squared'], label='train')
plt.plot(history.history['val_r_squared'], label='validation')
plt.grid(False)
plt.legend()
plt.title("R2")
plt.show()

## store the mse and r2 score and output the result
result = model.evaluate(test, test_labels)
print("mse: ", result[1])
print("r2: ", result[4])

## get the predicted values
y_predict = model.predict(test)

## store test labels and predicted labels in a dataframe
df = pd.DataFrame({"test_labels": test_labels, "y_predict": y_predict[:, 0]})
df.to_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/result/NN/CNN_yearly_human_result{}.csv".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))



