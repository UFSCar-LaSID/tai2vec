import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

import implicit
import scripts as kw
import gc

import time
import tensorflow as tf

import keras
from keras import layers, Model, Input, regularizers, initializers, callbacks
from scripts.recommenders.Item2vec.Data_repr import DataRepr
from scripts.recommenders.Item2vec.Item2Vec_abc import Item2vec_abstract
from keras.optimizers import Adam # type: ignore

class MemoryPrintingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
      gpu_dict = tf.config.experimental.get_memory_info('GPU:0')
      tf.print('\n GPU memory details [current: {} gb, peak: {} gb]'.format(
          float(gpu_dict['current']) / (1024 ** 3), 
          float(gpu_dict['peak']) / (1024 ** 3)))
      
class Item2vec_temp_aug_model(Item2vec_abstract):
    def __init__(self, embedding_dir, factors=100, w_size=-1, learning_rate=0.25, min_learning_rate = 0.000025, subsample = 0.0001, batch_size = kw.MEM_SIZE_LIMIT, negative_samples=5, negative_exp=0.75, epochs=100, time_exp=1, min_time_diff=300, lr_decay=0.95, regularization=-1):
        super().__init__(embedding_dir, factors, w_size, learning_rate, min_learning_rate, subsample, batch_size, negative_samples, negative_exp, epochs, lr_decay, regularization)
        self.time_exp = time_exp
        self.min_time_diff = min_time_diff

    def timestamp_diff(self, df):

        #Calcula a diferença de tempo entre a iteração atual e a passada, por usuário
        def calc_diff(df_group):

            df_group = df_group.sort_values(kw.COLUMN_DATETIME)
            df_group[kw.COLUMN_TIME_DIFF] = df_group[kw.COLUMN_DATETIME] - df_group[kw.COLUMN_DATETIME].shift(1)
            df_group[kw.COLUMN_TIME_DIFF] = df_group[kw.COLUMN_TIME_DIFF].fillna(pd.to_timedelta(0, unit='s'))
            df_group[kw.COLUMN_TIME_DIFF] = df_group[kw.COLUMN_TIME_DIFF].astype('int64')/ 10**9

            non_noise_diffs = df_group[df_group['timestamp_diff'] > self.min_time_diff]
            df_group['Q1'] = non_noise_diffs['timestamp_diff'].quantile(0.25)
            df_group['Q3'] = non_noise_diffs['timestamp_diff'].quantile(0.75)

            return df_group
        
        if kw.COLUMN_TIMESTAMP in df.columns:
            df[kw.COLUMN_DATETIME] = pd.to_datetime(df[kw.COLUMN_TIMESTAMP], unit='s')
        elif kw.COLUMN_DATETIME in df.columns:
            df[kw.COLUMN_DATETIME] = pd.to_datetime(df[kw.COLUMN_DATETIME])

        # Gera a coluna de diferença entre iterações
        df = df.groupby(kw.COLUMN_USER_ID, group_keys=False).apply(calc_diff).reset_index(drop=True)
        df[kw.COLUMN_THRESHOLD] = df['Q3'] + ((self.time_exp) * (df['Q3'] - df['Q1']))

        df['mask'] = df[kw.COLUMN_TIME_DIFF] >= df[kw.COLUMN_THRESHOLD]
        df['increment'] = df.groupby(kw.COLUMN_USER_ID)['mask'].cumsum()

        df.drop(columns=['mask'], inplace=True)

        return df

    def _generate_positive_data(self):

        X_target, X_context, sample_weights = [], [], []

        arr = np.arange(len(self.interaction_list))
        #np.random.shuffle(arr)

        for user_id in arr:

            X_target_aux, X_context_aux = [], []

            #Recebe a lista de itens do usuário atual
            user_time_groups = np.array(self.time_groups[user_id])
            curr_user = np.array(self.interaction_list[user_id])
            user_size = len(curr_user)
            
            if (user_size < 2):
                continue
                
            # Amostras positivas
            if self.window_size == -1:
                user_repeat = np.repeat(range(user_size), user_size-1)
                user_comb = np.tile(range(user_size), user_size)[np.tile(np.arange(1, user_size+1), user_size-1) + np.repeat(np.arange(user_size-1)*(user_size+1), user_size)]  

                X_target_aux.extend(curr_user[user_repeat])
                X_context_aux.extend(curr_user[user_comb])
                sample_values = np.where((user_time_groups[user_repeat] - user_time_groups[user_comb]) == 0, 2, 1)
                sample_weights.extend(sample_values)
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
                    sample_values = np.where((user_time_groups[i] - user_time_groups[context_indices]) == 0, 2, 1)
                    sample_weights.extend(sample_values)

            X_target.extend(X_target_aux)
            X_context.extend(X_context_aux)
            self.steps_per_epoch = (len(X_target) // self.batch_size) + 1

        return np.array(X_target), np.array(X_context), np.array(sample_weights)
            
    @tf.function
    def _generate_batches(self, target_items, positive_contexts, weights):

        batch_size = tf.shape(target_items)[0]

        target_items_repeated = tf.repeat(target_items, self.negative_samples + 1)
        #weight_repeated = tf.repeat(weights, self.negative_samples + 1)

        ones = tf.ones([tf.shape(weights)[0], self.negative_samples], dtype=weights.dtype)
        weight_repeated = tf.reshape(tf.concat([tf.expand_dims(weights, axis=1), ones], axis=1), [-1])
        
        #Seleciona os itens negativos
        random_samples = self.tf_generator.uniform(shape=(batch_size * self.negative_samples,), minval=0.0, maxval=1.0, dtype=tf.float32)
        negative_contexts = tf.searchsorted(self.cumulative_table, random_samples, side='right')
        negative_contexts = tf.reshape(negative_contexts, (batch_size, self.negative_samples))
        
        # Concatena o item positivo com o vetor de negativos
        positive_contexts = tf.expand_dims(positive_contexts, axis=1) 
        positive_contexts = tf.cast(positive_contexts, dtype=tf.int32)
        negative_contexts = tf.cast(negative_contexts, dtype=tf.int32)
        all_contexts = tf.concat([positive_contexts, negative_contexts], axis=1)
        
        # Define y = 1 para o item positivo e y = 0 para os negativos
        positive_labels = tf.ones((batch_size, 1), dtype=tf.float32)
        negative_labels = tf.zeros((batch_size, self.negative_samples), dtype=tf.float32)
        all_labels = tf.concat([positive_labels, negative_labels], axis=1)

        # Achata os contextos para corresponder aos target_items_repeated
        all_contexts_flat = tf.reshape(all_contexts, [-1])
        all_labels_flat = tf.reshape(all_labels, [-1])
        
        return (target_items_repeated, all_contexts_flat), all_labels_flat, weight_repeated
    
    def _data_generator(self, X_target, X_context_pos, sample_weights):

        dataset = tf.data.Dataset.from_tensor_slices((X_target, X_context_pos, sample_weights))
        dataset = dataset.batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self._generate_batches, num_parallel_calls=tf.data.AUTOTUNE)
        #dataset = dataset.cache()
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def fit(self, df):

        epochs_string = "@epochs={}".format(self.epochs)
        if os.path.exists(os.path.join(self.embedding_dir + epochs_string, kw.FILE_ITEMS_EMBEDDINGS)):
            return
        
        #Define os callbacks
        memory_printing_callback = MemoryPrintingCallback()
        epoch_callback = self.SaveEmbeddingsCallback(outer=self, save_interval=20)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=self.lr_decay, patience=3, min_lr=self.min_learning_rate, cooldown=5, verbose=1)
        
        np.random.seed(kw.RANDOM_STATE)
        tf.random.set_seed(kw.RANDOM_STATE)
        self.tf_generator = tf.random.Generator.from_seed(kw.RANDOM_STATE)

        if kw.COLUMN_TIMESTAMP in df.columns or kw.COLUMN_DATETIME in df.columns:
            df = self.timestamp_diff(df)
        else:
            raise Exception("Timestamp column not found")
        
        # Cria a representacao dos dados a partir do dataset
        self.data_repr = DataRepr(df)
        self.vocab_size = len(self.data_repr.le_items.classes_)

        # Reduz o dataset com subsampling e cria a lista de interações
        df = self._subsample_items(df)
        self.interaction_list = self.data_repr.create_interaction_list(df)

        sorted_df = df.sort_values(by=[kw.COLUMN_USER_ID, kw.COLUMN_DATETIME])
        self.time_groups = sorted_df.groupby(kw.COLUMN_USER_ID)['increment'].agg(list).to_list()

        #Cria a tabela cumulativa que será utilizada para o negative sampling
        self.item_freq = list(df.groupby(kw.COLUMN_ITEM_ID).size().values)
        self.cumulative_table = tf.constant(self._cumulative_table(self.item_freq), dtype=tf.float32)
                
        X_target, X_context_pos, sample_weights = self._generate_positive_data()
        data = self._data_generator(X_target, X_context_pos, sample_weights)

        self.model = self._build_model()

        self.model.fit(
            data, 
            epochs=self.epochs,
            shuffle=False,
            verbose=1, 
            callbacks=[epoch_callback],
        )
