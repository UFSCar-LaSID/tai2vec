import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, Model, initializers, regularizers, callbacks, Input
from keras.optimizers import Adam
import scripts as kw
from scripts.recommenders.Item2vec.Data_repr import DataRepr
import abc
import pickle
from keras.mixed_precision import set_global_policy
import torch
import torch.nn as nn

class Item2vec_abstract(abc.ABC):
    def __init__(self, embedding_dir, factors=100, w_size=-1, learning_rate=0.25, min_learning_rate = 0.000025 ,subsample = 0.001, 
                 batch_size = kw.MEM_SIZE_LIMIT, negative_samples=3, negative_exp=0.75, epochs=200, lr_decay=0.95, regularization=-1, recomender_norm=True):
        
        self.embedding_dir = embedding_dir
        self.embedding_size = factors
        self.window_size = w_size
        self.subsample_threshold = subsample
        self.negative_samples = negative_samples
        self.negative_expoent = negative_exp
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.regularization = regularization
        self.min_learning_rate = min_learning_rate

        self.X_target = []
        self.X_context = []
        self.y = []

        self.data_repr = None
        self.vocab_size = None
        self.subsample_probs = None
        self.model = None
        self.cumulative_table = None

    @abc.abstractmethod
    def fit(self, df):
        pass

    @property
    def items_embeddings(self) -> np.ndarray:
        return self.curr_model_item_embeddings

    def _subsample_items(self, df):

        freq = df.groupby(kw.COLUMN_ITEM_ID).size()
        n_interactions = len(df)
        z = freq / n_interactions
        keep_prob = (np.sqrt(z/self.subsample_threshold) + 1) * (self.subsample_threshold/z)
        keep_prob = keep_prob.reindex(df[kw.COLUMN_ITEM_ID])
        keep_prob.index = df.index
        discarded_interactions = keep_prob < np.random.rand(n_interactions)
        return df[~discarded_interactions].copy()
    
    def _cumulative_table(self, item_frequencies):

        # Cria a tabela cumulativa com todos os itens
        if self.negative_expoent > 0:
            item_frequencies = np.power(item_frequencies, self.negative_expoent)
        else:
            item_frequencies = np.reciprocal(np.power(item_frequencies, abs(self.negative_expoent)))

        total_count = np.sum(item_frequencies)
        probabilities = item_frequencies / total_count
        cum_table = np.cumsum(probabilities)
        return (cum_table / cum_table[-1])
    
    def get_item_embeddings(self) -> np.ndarray:
        return self.curr_model_item_embeddings