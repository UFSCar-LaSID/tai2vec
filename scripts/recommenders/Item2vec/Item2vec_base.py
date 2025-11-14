import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# PyTorch imports
import torch
import torch.nn as nn

import scripts as kw
from .Data_repr import DataRepr
from .Item2Vec_abc import Item2vec_abstract
from .torchmodules.pytorch_dataset import Item2VecDataset
from .torchmodules.pytorch_trainer import Item2VecTrainer
from .torchmodules.pytorch_model import Item2VecModel
from .torchmodules.pytorch_dataset import create_item2vec_dataloader
from scripts.recommenders.utils.monitor import monitor

      
class Item2vec_model(Item2vec_abstract):

    def __init__(self, embedding_dir, factors=100, w_size=-1, learning_rate=0.25, min_learning_rate=0.000025,
                 subsample=0.001, batch_size=kw.MEM_SIZE_LIMIT, negative_samples=3, negative_exp=0.75,
                 epochs=100, lr_decay=0.95, regularization=-1, recomender_norm=True):
        
        super().__init__(embedding_dir, factors, w_size, learning_rate, min_learning_rate,
                         subsample, batch_size, negative_samples, negative_exp, epochs, lr_decay, regularization)
        
        print("Using Item2Vec model with parameters:")
        print(self.__dict__)
        
        self.interaction_matrix = None
        self.user_lengths = None

    def _generate_positive_data(self):
        X_target = []
        X_context = []

        arr = np.arange(len(self.interaction_list))
        np.random.shuffle(arr)

        for user_id in arr:
            curr_user = np.array(self.interaction_list[user_id], dtype=np.int32)
            user_size = curr_user.shape[0]

            if user_size < 2:
                continue

            # Sliding window positive sampling only
            for i in range(user_size):
                start_idx = max(0, i - self.window_size)
                end_idx = min(user_size, i + self.window_size + 1)

                # Context indices excluding target itself
                context_indices = np.arange(start_idx, end_idx)
                context_indices = context_indices[context_indices != i]

                if len(context_indices) == 0:
                    continue

                # Expand pairs
                X_target.extend([curr_user[i]] * len(context_indices))
                X_context.extend(curr_user[context_indices])

        X_target = np.asarray(X_target, dtype=np.int32)
        X_context = np.asarray(X_context, dtype=np.int32)

        print(f"\nNumber of positive samples: {len(X_target):,}")
        print(f"Number of negative samples: {len(X_target) * self.negative_samples:,}")

        # Compute steps per epoch once
        self.steps_per_epoch = (len(X_target) // self.batch_size) + (
            1 if len(X_target) % self.batch_size else 0
        )

        return X_target, X_context

    @monitor
    def fit(self, df):

        if os.path.exists(self.embedding_dir + "@epochs=" + str(self.epochs)):
            return

        np.random.seed(kw.RANDOM_STATE)
        torch.manual_seed(kw.RANDOM_STATE)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(kw.RANDOM_STATE)

        df = self._subsample_items(df)
        self.data_repr = DataRepr(df)
        self.interaction_list = self.data_repr.create_interaction_list(df)
        self.item_freq = list(df.groupby(kw.COLUMN_ITEM_ID).size().values)
        self.cumulative_table = self._cumulative_table(self.item_freq)
        X_target, X_context = self._generate_positive_data()
        max_workers = os.cpu_count()
        print(f"Using {max_workers} workers for DataLoader")
        
        self.model = Item2VecModel(
            vocab_size=df[kw.COLUMN_ITEM_ID].nunique(), 
            embedding_size=self.embedding_size, 
            learning_rate=self.learning_rate, 
            lr_decay=self.lr_decay, 
            regularization=self.regularization,
            loss_sum=True
        ).to('cuda' if torch.cuda.is_available() else 'cpu')
        
        dataloader = create_item2vec_dataloader(
            X_target=X_target, 
            X_context=X_context, 
            batch_size=self.batch_size, 
            weights=None, 
            shuffle=False,
            num_workers=8,
            cumulative_table=self.cumulative_table, 
            negative_samples=self.negative_samples
        )

        trainer = Item2VecTrainer(self, self.model)
        self.model = trainer.train(dataloader, data_repr=self.data_repr)