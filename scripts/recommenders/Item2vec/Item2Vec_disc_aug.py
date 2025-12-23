import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# Disable mixed precision and XLA for stability with large datasets
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import scripts as kw

# PyTorch imports
import torch
import torch.nn as nn

from .Data_repr import DataRepr
from .Item2Vec_abc import Item2vec_abstract
from .torchmodules.pytorch_dataset import Item2VecDataset
from .torchmodules.pytorch_trainer import Item2VecTrainer
from .torchmodules.pytorch_model import Item2VecModel
from .torchmodules.pytorch_dataset import create_item2vec_dataloader

      
class Item2vec_temp_aug_model(Item2vec_abstract):
    def __init__(self, embedding_dir, factors=100, w_size=-1, learning_rate=0.25, min_learning_rate = 0.000025, subsample = 0.001, batch_size = kw.MEM_SIZE_LIMIT, negative_samples=3, 
                 negative_exp=0.75, epochs=100, lr_decay=0.95, regularization=-1, recomender_norm=True, time_exp=1.5, min_time_diff=300, big_innit=False):
        super().__init__(embedding_dir, factors, w_size, learning_rate, min_learning_rate, subsample, batch_size, negative_samples, negative_exp, epochs, lr_decay, regularization, recomender_norm, big_innit)
        self.time_exp = time_exp
        self.min_time_diff = min_time_diff

    def timestamp_diff(self, df):

        # 1. Handle Datetime conversion
        if kw.COLUMN_TIMESTAMP in df.columns:
            df = df.copy()
            df[kw.COLUMN_DATETIME] = pd.to_datetime(df[kw.COLUMN_TIMESTAMP], unit='s')
        elif kw.COLUMN_DATETIME in df.columns:
            df = df.copy()
            df[kw.COLUMN_DATETIME] = pd.to_datetime(df[kw.COLUMN_DATETIME])

        df = df.sort_values([kw.COLUMN_USER_ID, kw.COLUMN_DATETIME])

        # 3. Calculate Time Diffs
        df[kw.COLUMN_TIME_DIFF] = df.groupby(kw.COLUMN_USER_ID)[kw.COLUMN_DATETIME].diff().dt.total_seconds().fillna(0).astype('int32')

        # 4. Filter Valid Gaps (Optimization)
        valid_gaps_mask = df[kw.COLUMN_TIME_DIFF] > self.min_time_diff
        df['temp_diffs'] = df[kw.COLUMN_TIME_DIFF].where(valid_gaps_mask)

        # 5. Calculate Quantiles efficiently
        q1 = df.groupby(kw.COLUMN_USER_ID)['temp_diffs'].transform('quantile', 0.25)
        q3 = df.groupby(kw.COLUMN_USER_ID)['temp_diffs'].transform('quantile', 0.75)

        # 6. Calculate Threshold
        iqr = q3 - q1
        threshold = q3 + (self.time_exp * iqr)

        # 7. Safety Fix: Fill NaN thresholds with Infinity
        threshold = threshold.fillna(np.inf)

        # 8. Create Session Mask
        df['mask'] = df[kw.COLUMN_TIME_DIFF] >= threshold
        df['increment'] = df.groupby(kw.COLUMN_USER_ID)['mask'].cumsum()

        # 9. Cleanup
        df.drop(columns=['mask', 'temp_diffs'], inplace=True)

        return df

    def _generate_positive_data(self):
        X_target = []
        X_context = []
        sample_weights = []

        for user_id in range(len(self.interaction_list)):
            curr_user = np.array(self.interaction_list[user_id], dtype=np.int32)
            user_time_groups = np.array(self.time_groups[user_id], dtype=np.int32)
            user_size = curr_user.shape[0]

            if user_size < 2:
                continue

            for i in range(user_size):
                start_idx = max(0, i - self.window_size)
                end_idx = min(user_size, i + self.window_size + 1)

                # Context items (excluding the target itself)
                context_indices = np.arange(start_idx, end_idx)
                context_indices = context_indices[context_indices != i]

                # Targets repeated for each context
                X_target.extend([curr_user[i]] * len(context_indices))
                X_context.extend(curr_user[context_indices])

                # Time-based weighting: 2 if same time group, else 1
                weights = np.where(user_time_groups[i] == user_time_groups[context_indices], 2.0, 1.0)

                sample_weights.extend(weights)

        # Convert to arrays once at the end
        X_target = np.asarray(X_target, dtype=np.int32)
        X_context = np.asarray(X_context, dtype=np.int32)
        sample_weights = np.asarray(sample_weights, dtype=np.float32)

        # Compute steps per epoch once
        self.steps_per_epoch = (len(X_target) // self.batch_size) + (
            1 if len(X_target) % self.batch_size else 0)

        return X_target, X_context, sample_weights


    def _fit_data(self, df):

        self.item_freq = list(df.groupby(kw.COLUMN_ITEM_ID).size().values)
        self.cumulative_table = self._cumulative_table(self.item_freq)
        self.vocab_size = len(self.item_freq)

        if kw.COLUMN_TIMESTAMP in df.columns or kw.COLUMN_DATETIME in df.columns:
            df = self.timestamp_diff(df.copy())
        else:
            raise Exception("Timestamp column not found")
        
        self.data_repr = DataRepr(df)
        df = self._subsample_items(df)
        self.interaction_list = self.data_repr.create_interaction_list(df)

        sorted_df = df.sort_values(by=[kw.COLUMN_USER_ID, kw.COLUMN_DATETIME])
        self.time_groups = sorted_df.groupby(kw.COLUMN_USER_ID)['increment'].agg(list).to_list()
                
        X_target, X_context, sample_weights = self._generate_positive_data()

        self.model = Item2VecModel(
            vocab_size=self.vocab_size,
            embedding_size=self.embedding_size, 
            learning_rate=self.learning_rate, 
            lr_decay=self.lr_decay, 
            regularization=self.regularization,
            loss_sum=True,
            big_innit=self.big_innit
        ).to('cuda' if torch.cuda.is_available() else 'cpu')

        max_workers = os.cpu_count()
        print(f"Using {max_workers} workers for DataLoader")
        
        dataloader = create_item2vec_dataloader(
            X_target=X_target, 
            X_context=X_context, 
            cumulative_table=self.cumulative_table, 
            negative_samples=self.negative_samples,
            batch_size=self.batch_size, 
            weights=sample_weights, 
            shuffle=True,
            num_workers=8
        )

        trainer = Item2VecTrainer(self, self.model)
        self.model = trainer.train(dataloader, self.data_repr)
        
        return self.model.get_item_embeddings()