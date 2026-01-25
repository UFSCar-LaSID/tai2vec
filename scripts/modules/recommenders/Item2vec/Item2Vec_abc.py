from abc import ABC
import numpy as np
import os
import pickle
import scripts as kw
import abc
import pickle
import torch

class Item2vec_abstract(abc.ABC):

    def __init__(self, embedding_dir, factors=100, w_size=-1, learning_rate=0.25, min_learning_rate = 0.000025 ,subsample = 0.001, 
                 batch_size = kw.MEM_SIZE_LIMIT, negative_samples=3, negative_exp=0.75, epochs=200, lr_decay=0.95, regularization=-1, recomender_norm=True, big_innit=False):
        
        self.embedding_dir = embedding_dir
        self.embedding_size = 50
        self.window_size = w_size
        self.subsample_threshold = subsample
        self.negative_samples = 7
        self.negative_expoent = negative_exp
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = 2**14
        self.lr_decay = 0.0001
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
        self.item_freq = None
        self.big_innit = big_innit

    @abc.abstractmethod
    def _fit_data(self, df):
        raise NotImplementedError

    def fit(self, df):
        path = self.embedding_dir
        if os.path.exists(os.path.join(path, kw.FILE_ITEMS_EMBEDDINGS)):
            print(f"Embeddings already exist at {os.path.join(path, kw.FILE_ITEMS_EMBEDDINGS)}, skipping training.")
            with open(os.path.join(path, kw.FILE_SPARSE_REPR), 'rb') as f:
                self.data_repr = pickle.load(f)
            return

        np.random.seed(kw.RANDOM_STATE)
        torch.manual_seed(kw.RANDOM_STATE)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(kw.RANDOM_STATE)

        self.item_freq = list(df.groupby(kw.COLUMN_ITEM_ID).size().values)
        self.cumulative_table = self._cumulative_table(self.item_freq)
        self.vocab_size = len(self.item_freq)

        self._fit_data(df)

        self._save_embeddings()

    def _save_embeddings(self):
        
        embeddings_target, embeddings_context = self.get_item_embeddings()

        path = self.embedding_dir# + "@epochs=" + str(self.epochs)
        os.makedirs(path, exist_ok=True)

        np.save(os.path.join(path, kw.FILE_ITEMS_EMBEDDINGS), embeddings_target)
        np.save(os.path.join(path, kw.FILE_CONTEXT_EMBEDDINGS), embeddings_context)

        with open(os.path.join(path, kw.FILE_SPARSE_REPR), 'wb') as f:
            pickle.dump(self.data_repr, f)

    def get_item_embeddings(self):
        return self.model.get_item_embeddings()

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

        if self.negative_expoent > 0:
            item_frequencies = np.power(item_frequencies, self.negative_expoent)
        else:
            item_frequencies = np.reciprocal(np.power(item_frequencies, abs(self.negative_expoent)))

        total_count = np.sum(item_frequencies)
        probabilities = item_frequencies / total_count
        cum_table = np.cumsum(probabilities)
        return (cum_table / cum_table[-1])