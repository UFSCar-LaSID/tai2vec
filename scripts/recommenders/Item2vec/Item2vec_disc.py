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
from scripts.recommenders.utils import monitor

      
class Item2vec_temp_model(Item2vec_abstract):

    def __init__(self, embedding_dir, factors=50, w_size=-1, learning_rate=0.25, min_learning_rate = 0.0025 ,subsample = 0.001, batch_size = kw.MEM_SIZE_LIMIT, negative_samples=3, negative_exp=0.75, epochs=100, lr_decay=0.1, time_exp=1.5, min_time_diff=300, regularization=-1, init_strat='uniform_small', recomender_norm=True):
        super().__init__(embedding_dir, factors, w_size, learning_rate, min_learning_rate, subsample, batch_size, negative_samples, negative_exp, epochs, lr_decay, regularization, init_strat)
        self.time_exp = time_exp
        self.min_time_diff = min_time_diff

    def timestamp_diff(self, df):

        if kw.COLUMN_TIMESTAMP in df.columns:
            df[kw.COLUMN_DATETIME] = pd.to_datetime(df[kw.COLUMN_TIMESTAMP], unit='s')
        elif kw.COLUMN_DATETIME in df.columns:
            df[kw.COLUMN_DATETIME] = pd.to_datetime(df[kw.COLUMN_DATETIME])

        df[kw.COLUMN_TIME_DIFF] = df.groupby(kw.COLUMN_USER_ID)[kw.COLUMN_DATETIME].diff().dt.total_seconds().fillna(0).astype('int32')
    
        q1 = df.groupby(kw.COLUMN_USER_ID)[kw.COLUMN_TIME_DIFF].transform(lambda x: x[x > self.min_time_diff].quantile(0.25) if (x > self.min_time_diff).any() else np.inf)
        q3 = df.groupby(kw.COLUMN_USER_ID)[kw.COLUMN_TIME_DIFF].transform(lambda x: x[x > self.min_time_diff].quantile(0.75) if (x > self.min_time_diff).any() else np.inf)
        threshold = q3 + (self.time_exp * (q3 - q1))
        
        df['mask'] = df[kw.COLUMN_TIME_DIFF] >= threshold
        df['increment'] = df.groupby(kw.COLUMN_USER_ID)['mask'].cumsum()
        df['old_user_id'] = df[kw.COLUMN_USER_ID]
        df[kw.COLUMN_USER_ID] = df.groupby([kw.COLUMN_USER_ID, 'increment']).ngroup()
        
        return df

    def _generate_positive_data(self):

        X_target, X_context = [], []

        arr = np.arange(len(self.interaction_list))
        np.random.shuffle(arr)

        for user_id in arr:

            X_target_aux, X_context_aux = [], []

            #Recebe a lista de itens do usuário atual
            curr_user = self.interaction_list[user_id]
            np.random.shuffle(curr_user)
            user_size = len(curr_user)
            
            if (user_size < 2):
                continue
                
            # Amostras positivas
            if self.window_size == -1:
                X_target_aux.extend(np.repeat(curr_user, user_size-1))
                X_context_aux.extend(np.tile(curr_user, user_size)[np.tile(np.arange(1, user_size+1), user_size-1) + np.repeat(np.arange(user_size-1)*(user_size+1), user_size)])
            else:
                for i in range(user_size):
                    #Define o início e o fim da janela de contexto
                    start_idx = max(0, i - self.window_size)
                    end_idx = min(user_size, i + self.window_size + 1)
                    # Cria um array de indices e remove o alvo
                    context_indices = np.arange(start_idx, end_idx)
                    context_indices = context_indices[context_indices != i] 
                    # Calcula os ids positivos
                    X_target_aux.extend(np.repeat(curr_user[i], len(context_indices)))
                    X_context_aux.extend(np.array(curr_user)[context_indices])

            X_target.extend(X_target_aux)
            X_context.extend(X_context_aux)

        print("\nNumber of samples:", len(X_target))
        print("Number of negative samples:", len(X_target) * self.negative_samples)
        self.steps_per_epoch = (len(X_target) // self.batch_size) + 1

        return np.array(X_target), np.array(X_context)

    def fit(self, df):

        if os.path.exists(self.embedding_dir + "@epochs=5"):
            return

        np.random.seed(kw.RANDOM_STATE)
        torch.manual_seed(kw.RANDOM_STATE)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(kw.RANDOM_STATE)

        if kw.COLUMN_TIMESTAMP in df.columns or kw.COLUMN_DATETIME in df.columns:
            df = self.timestamp_diff(df.copy())
        else:
            raise Exception("Timestamp column not found")

        df = self._subsample_items(df)
        self.data_repr = DataRepr(df)
        self.interaction_list = self.data_repr.create_interaction_list(df)
        
        self.model = Item2VecModel(
            vocab_size=df[kw.COLUMN_ITEM_ID].nunique(), 
            embedding_size=self.embedding_size, 
            learning_rate=self.learning_rate, 
            lr_decay=self.lr_decay, 
            regularization=self.regularization,
            init_strat=self.init_strat
        ).to('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.item_freq = list(df.groupby(kw.COLUMN_ITEM_ID).size().values)
        self.cumulative_table = self._cumulative_table(self.item_freq)
        
        X_target, X_context = self._generate_positive_data()

        max_workers = os.cpu_count()
        print(f"Using {max_workers} workers for DataLoader")
        
        dataloader = create_item2vec_dataloader(
            X_target=X_target, 
            X_context=X_context, 
            cumulative_table=self.cumulative_table, 
            negative_samples=self.negative_samples,
            batch_size=self.batch_size, 
            weights=None, 
            shuffle=False,
            num_workers=max_workers
        )

        trainer = Item2VecTrainer(self, self.model)
        self.model = trainer.train(dataloader, data_repr=self.data_repr)
    