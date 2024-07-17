#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a neural network to predict the number of WNV infected mosquitos in a given area.
"""
from elwnv.corpus import Corpus
from elwnv.mlp import MLPModel
from elwnv.plot import Plot
from hyperopt import fmin, tpe, hp

corpus = Corpus()
mlp = MLPModel(corpus)

plot = Plot()
mlp.history_df.loc[:, ['loss', 'val_loss']].plot(title="MSE", logy=True)
# plot.sensitivity(
#     train = corpus.train,
#     test = corpus.test,
#     model = mlp.model,
#     test_column_names=corpus.test_column_names,
# )

plot.show()
