from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import pandas as pd
from tensorflow_addons.metrics import RSquare


class MLPModel:
    def __init__(self, corpus):
        # Define the input layers for land use class
        land_use = layers.Input(shape=(1,), name="cat_land_use_class")

        # Define the input layers for land change
        land_change = layers.Input(shape=(1,), name="cat_land_change")

        # Define the input layers for non-categorical features and add batch normalization
        noncat = layers.Input(shape=corpus.input_non_categorical_shape, name="noncat")

        net1 = self.inner_model(corpus, land_use, land_change, noncat)
        pipe = net1
        # net2 = self.inner_model(corpus, land_use, land_change, noncat)
        # net3 = self.inner_model(corpus, land_use, land_change, noncat)
        # net4 = self.inner_model(corpus, land_use, land_change, noncat)
        # pipe = (net1 + net2 + net3 + net4) / 4

        # use exponential activation function for regression to shrink big numbers of target "COUNT"
        pipe = layers.Dense(1, activation=tf.exp)(pipe)


        # pipe = layers.Dense(1)(pipe)
        # build model
        self.model = models.Model(inputs=[land_use, land_change, noncat], outputs=[pipe])

        ## compile model using regression metrics
        ## using 2017 horse data as prediction compare to 2018 as real horse data
        ## the RMSE is 0.9326036902362 and STD is 1.0621133906210167
        ##
        self.model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae", "mse"],
            # metrics=[RSquare()],
        )

        # early stopping
        early_stopping = callbacks.EarlyStopping(
            patience=20,
            min_delta=0.001,
            restore_best_weights=True,
        )

        # first adapt normalization layer for preprocessing data
        # norm.adapt(train)

        input_train_dict = {"cat_land_use_class": corpus.input_land_use_train, "cat_land_change": corpus.land_change_train, "noncat": corpus.train}
        input_test_dict = {"cat_land_use_class": corpus.input_land_use_test, "cat_land_change": corpus.land_change_test, "noncat": corpus.test}


        # fit model
        history = self.model.fit(
            input_train_dict,
            corpus.train_labels,
            validation_data=(input_test_dict, corpus.test_labels),
            batch_size=32,
            epochs=1000,
            # sample_weight=corpus.sample_weights,
            callbacks=[early_stopping],
        )

        ## save the best model
        self.model.save("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results/"
                        "models/model_weights_{}" .format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))

        # preds = pd.DataFrame(dict(
        #     pred_wnv=self.model.predict(input_test_dict)[:, -1],
        #     actual_wnv=corpus.test_labels,
        # )).sort_values("actual_wnv", ascending=False)

        # #unnormalize the prediction
        epslon = 1e-6
        preds = pd.DataFrame(dict(
            pred_wnv=self.model.predict(input_test_dict)[:, -1] * corpus.train_labels_std + corpus.train_labels_mean - epslon,
            actual_wnv=corpus.test_labels * corpus.test_labels_std + corpus.test_labels_mean - epslon,
        )).sort_values("actual_wnv", ascending=False)

        breakpoint()

        preds.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/results"
                     "/prediction_results_pca.csv", index=True)

        self.history_df = pd.DataFrame(history.history)

    @staticmethod
    def inner_model(corpus, land_use, land_change, noncat):
        num_nodes = 30

        # embedding layer for Species
        pipe1 = layers.Embedding(corpus.unique_categorical_land_use.size, num_nodes)(land_use)
        pipe1 = tf.reshape(pipe1, [-1, num_nodes])

        # embedding layer for land change
        pipe2 = layers.Embedding(corpus.unique_categorical_land_change_shape.size, num_nodes)(land_change)
        pipe2 = tf.reshape(pipe2, [-1, num_nodes])

        pipe = layers.BatchNormalization()(noncat)
        # Create the hidden layer the rest of the neural network layers
        pipe = layers.Dense(num_nodes, activation='relu')(pipe)
        # # use LeakyReLU to avoid dead neuron
        # pipe = layers.Dense(4)(pipe)
        # pipe = layers.LeakyReLU()(pipe)

        pipe = pipe + pipe1 + pipe2
        pipe = layers.BatchNormalization()(pipe)
        pipe = layers.GaussianDropout(0.3)(pipe)

        pipe = layers.Dense(num_nodes, activation="relu")(pipe)
        # # use LeakyReLU to avoid dead neuron
        # pipe = layers.Dense(4)(pipe)
        # pipe = layers.LeakyReLU()(pipe)
        pipe = layers.BatchNormalization()(pipe)
        pipe = layers.GaussianDropout(0.3)(pipe)

        pipe = layers.Dense(num_nodes, activation="relu")(pipe)
        # # use LeakyReLU to avoid dead neuron
        # pipe = layers.Dense(4)(pipe)
        # pipe = layers.LeakyReLU()(pipe)
        pipe = layers.BatchNormalization()(pipe)
        pipe = layers.GaussianDropout(0.3)(pipe)

        return pipe
