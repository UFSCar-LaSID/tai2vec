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
from sklearn.preprocessing import MinMaxScaler

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
      
class Item2vec_Temp_Cont_model(Item2vec_abstract):
    def __init__(self, embedding_dir, factors=100, w_size=-1, learning_rate=0.25, min_learning_rate = 0.0025, subsample = 0.0001, batch_size = kw.MEM_SIZE_LIMIT, negative_samples=5, negative_exp=0.75, min_weight = 0.3, curve_exp = 2, epochs=100, min_time_diff=60, weight_floor=0.3, lr_decay=0.96, regularization=-1):
        super().__init__(embedding_dir, factors, w_size, learning_rate, min_learning_rate, subsample, batch_size, negative_samples, negative_exp, epochs, lr_decay, regularization)
        self.min_time_diff = min_time_diff
        self.min_weight = min_weight
        self.curve_exp = curve_exp
        self.weight_floor = weight_floor
    
    def timestamp_cum(self, df):
        """
        Fast timestamp cumulative calculation using vectorized operations with scaler.
        """
        # Handle datetime conversion
        if kw.COLUMN_TIMESTAMP in df.columns:
            df = df.copy()
            df[kw.COLUMN_DATETIME] = pd.to_datetime(df[kw.COLUMN_TIMESTAMP], unit='s')
        elif kw.COLUMN_DATETIME in df.columns:
            df = df.copy()
            df[kw.COLUMN_DATETIME] = pd.to_datetime(df[kw.COLUMN_DATETIME])
        
        # Sort once
        df = df.sort_values([kw.COLUMN_USER_ID, kw.COLUMN_DATETIME])
        
        # Calculate time differences
        df[kw.COLUMN_TIME_DIFF] = (
            df.groupby(kw.COLUMN_USER_ID)[kw.COLUMN_DATETIME]
            .diff()
            .dt.total_seconds()
            .fillna(0)
        )
        
        # Create mask for valid differences (non-noise)
        valid_mask = df[kw.COLUMN_TIME_DIFF] > self.min_time_diff
        
        # Create a temporary column with only valid differences
        df['valid_diffs'] = df[kw.COLUMN_TIME_DIFF].where(valid_mask)
        
        # Calculate Q1, Q3 and IQR using transform (vectorized)
        q1 = df.groupby(kw.COLUMN_USER_ID)['valid_diffs'].transform('quantile', 0.25).fillna(0)
        q3 = df.groupby(kw.COLUMN_USER_ID)['valid_diffs'].transform('quantile', 0.75).fillna(0)
        iqr = q3 - q1
        
        # Calculate mean and std from valid diffs only (vectorized)
        df[kw.COLUMN_MEAN] = df.groupby(kw.COLUMN_USER_ID)['valid_diffs'].transform('mean').fillna(0)
        df[kw.COLUMN_STD] = df.groupby(kw.COLUMN_USER_ID)['valid_diffs'].transform('std').fillna(0)
        
        # Clip time_diff using Q3 + 1.5 * IQR (vectorized)
        threshold = 1.5
        upper_clip = q3 + (threshold * iqr)
        df[kw.COLUMN_TIME_DIFF] = df[kw.COLUMN_TIME_DIFF].clip(upper=upper_clip)
        
        # Calculate cumulative sum per user (vectorized)
        df[kw.COLUMN_TIME_CUMSUM] = df.groupby(kw.COLUMN_USER_ID)[kw.COLUMN_TIME_DIFF].cumsum()
        
        # Apply MinMaxScaler normalization per user group using transform
        def scale_group(group):
            if len(group) == 1:
                return pd.Series([self.min_weight], index=group.index)
            
            scaler = MinMaxScaler(feature_range=(self.min_weight, 1))
            scaled = scaler.fit_transform(group.values.reshape(-1, 1)).flatten()
            return pd.Series(scaled, index=group.index)
        
        df[kw.COLUMN_TIME_CUMSUM_NORM] = (
            df.groupby(kw.COLUMN_USER_ID)[kw.COLUMN_TIME_CUMSUM]
            .transform(scale_group)
        )
        
        # Clean up temporary column
        df = df.drop(columns=['valid_diffs'])
        
        return df

    def _calculate_weights(self, distances, mean, std): 

        def z_scores(distances, mean, std):
            if std == 0:
                return np.zeros(len(distances))
            return (distances - mean) / std
        
        z_score = z_scores(distances, mean, std)
        z_score = np.clip(z_score, 0, 20)
                
        #print("Z-Score: ", np.round(z_score, 2))
        weights = np.where(z_score == 0, 1, 1 - np.power((z_score / (z_score + 1)), self.curve_exp))
        #print("Weights: ", np.round(weights, 2))
        return np.round(weights, 2)

    def _generate_positive_data(self):

        X_target, X_context, y, sample_weights = [], [], [], []

        arr = np.arange(len(self.interaction_list))
        #np.random.shuffle(arr)

        for user_id in arr:

            X_target_aux, X_context_aux, y_aux = [], [], [] 

            #Recebe a lista de itens do usuário atual
            curr_user = np.array(self.interaction_list[user_id])
            cumulative_time = np.array(self.cumsum_list[user_id])
            norm_weights = np.array(self.norm_weight_list[user_id])

            mean = self.mean_list[user_id]
            std = self.std_list[user_id]
            user_size = len(curr_user)

            if (user_size < 2):
                continue
                
            # Amostras positivas
            if self.window_size == -1:
                
                user_repeat = np.repeat(range(user_size), user_size-1)
                user_comb = np.tile(range(user_size), user_size)[np.tile(np.arange(1, user_size+1), user_size-1) + np.repeat(np.arange(user_size-1)*(user_size+1), user_size)]  

                X_target_aux.extend(curr_user[user_repeat])
                X_context_aux.extend(curr_user[user_comb])
                norm_sample_weights_aux = 1 - abs(norm_weights[user_repeat] - norm_weights[user_comb])
                norm_sample_weights = np.round(np.maximum(1 - (np.log10(1/norm_sample_weights_aux)), self.weight_floor), 2)
                if self.curve_exp == -1:
                    sample_weights.extend(norm_sample_weights)
                else:
                    z_sample_weights = self._calculate_weights(abs(cumulative_time[user_repeat] - cumulative_time[user_comb]), mean, std)
                    final_weights = (z_sample_weights + norm_sample_weights)/2
                    sample_weights.extend(final_weights)
            else:
                for i in range(user_size):
                    #Define o início e o fim da janela de contexto
                    start_idx = max(0, i - self.window_size)
                    end_idx = min(user_size, i + self.window_size + 1)
                    # Cria um array de indices e remove o alvo
                    context_indices = np.arange(start_idx, end_idx)
                    # Calcula os ids positivos
                    X_target_aux.extend(np.repeat(curr_user[i], len(context_indices)))
                    X_context_aux.extend(np.array(curr_user)[context_indices])
                    norm_sample_weights_aux = 1 - abs(norm_weights[i] - norm_weights[context_indices])
                    norm_sample_weights = np.round(np.maximum(1 - (np.log10(1/norm_sample_weights_aux)), self.weight_floor), 2)
                    if self.curve_exp == -1:
                        sample_weights.extend(norm_sample_weights)
                    else:
                        z_sample_weights = self._calculate_weights(abs(cumulative_time[i] - cumulative_time[context_indices]), mean, std)
                        #final_weights = (z_sample_weights + norm_sample_weights)/2
                        sample_weights.extend(z_sample_weights)

            X_target.extend(X_target_aux)
            X_context.extend(X_context_aux)
            
        print("\nNumber of samples:", len(X_target))
        print("Number of negative samples:", len(X_target) * self.negative_samples)
        self.steps_per_epoch = (len(X_target) // self.batch_size) + 1

        return np.array(X_target), np.array(X_context), np.array(sample_weights)

    def fit(self, df):
        
        np.random.seed(kw.RANDOM_STATE)
        torch.manual_seed(kw.RANDOM_STATE)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(kw.RANDOM_STATE)

        if kw.COLUMN_TIMESTAMP in df.columns or kw.COLUMN_DATETIME in df.columns:
            df = self.timestamp_cum(df.copy())
        else:
            raise Exception("Timestamp column not found")

        # Reduz o dataset com subsampling e cria a lista de interações e de pesos
        df = self._subsample_items(df)

        # Cria a representacao dos dados a partir do dataset
        self.data_repr = DataRepr(df, temporal_sorting=True)

        # Cria as listas de interações e pesos
        self.interaction_list = self.data_repr.create_column_list(df, kw.COLUMN_ITEM_ID)
        self.cumsum_list = self.data_repr.create_column_list(df, kw.COLUMN_TIME_CUMSUM)
        self.norm_weight_list = self.data_repr.create_column_list(df, kw.COLUMN_TIME_CUMSUM_NORM)
        self.mean_list = self.data_repr.create_metrics_list(df, kw.COLUMN_MEAN)
        self.std_list = self.data_repr.create_metrics_list(df, kw.COLUMN_STD)

        #Cria a tabela cumulativa que será utilizada para o negative sampling
        self.item_freq = list(df.groupby(kw.COLUMN_ITEM_ID).size().values)
        self.cumulative_table = self._cumulative_table(self.item_freq)

        X_target, X_context, sample_weights = self._generate_positive_data()

        # Calculate vocab_size to accommodate the full range of item IDs
        vocab_size = df[kw.COLUMN_ITEM_ID].max() + 1

        self.model = Item2VecModel(
            vocab_size=df[kw.COLUMN_ITEM_ID].nunique(), 
            embedding_size=self.embedding_size, 
            learning_rate=self.learning_rate, 
            lr_decay=self.lr_decay, 
            regularization=self.regularization
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
            shuffle=False,
            num_workers=max_workers
        )

        trainer = Item2VecTrainer(self, self.model)
        self.model = trainer.train(dataloader, self.data_repr)
        