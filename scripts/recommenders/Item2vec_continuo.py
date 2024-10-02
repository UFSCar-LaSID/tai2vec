import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

import implicit
import scripts as kw

import time

import keras
from keras import layers, Model, Input, regularizers, initializers
from keras.optimizers import Adam # type: ignore

class DataRepr(object):
    
    def __init__(self, df):
        self.le_users = LabelEncoder()
        self.le_users.fit(df['id_user'])
        self.le_items = LabelEncoder()
        self.le_items.fit(df['id_item'])
        self.interaction_matrix = self.create_user_items_matrix(df)
    
    def create_user_items_matrix(self, df):
        data = df[kw.COLUMN_TIME_CUMSUM].to_list()
        user_ind = self.le_users.transform(df['id_user'])
        item_ind = self.le_items.transform(df['id_item'])
        n_users = len(self.le_users.classes_)
        n_items = len(self.le_items.classes_)
        return csr_matrix((data, (user_ind, item_ind)), shape=(n_users, n_items))
    
    def get_user_index(self, user_id):
        return self.le_users.transform([user_id])[0]
    
    def get_item_index(self, item_id):
        return self.le_items.transform([item_id])[0]
    
    def get_user_id(self, user_index):
        return self.le_users.inverse_transform(user_index)
    
    def get_item_id(self, item_index):
        return self.le_items.inverse_transform(item_index)
    
    def get_user_items_matrix(self):
        return self.interaction_matrix
    
    def get_interaction_list(self):
        
        # Remove itens negativos
        user_indices, item_indices = self.interaction_matrix.nonzero()
        
        # Agrupa os usuários e gera a representação densa
        users, user_pos = np.unique(user_indices, return_index=True)
        interaction_list = np.split(item_indices, user_pos[1:])
        interaction_list = [list(items) for items in interaction_list]
        
        return interaction_list

class MemoryPrintingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
      gpu_dict = tf.config.experimental.get_memory_info('GPU:0')
      tf.print('\n GPU memory details [current: {} gb, peak: {} gb]'.format(
          float(gpu_dict['current']) / (1024 ** 3), 
          float(gpu_dict['peak']) / (1024 ** 3)))

class Item2vec_Temp_Cont_model:
    def __init__(self, embedding_dir, factors=128, w_size=-1, learning_rate=0.25, subsample = 0.0001, batch_size = kw.MEM_SIZE_LIMIT, negative_samples=5, negative_exp=0.75, epochs=5):
        
        self.embedding_dir = embedding_dir
        self.embedding_size = factors
        self.window_size = w_size
        self.subsample_threshold = subsample
        self.negative_samples = negative_samples
        self.negative_expoent = negative_exp
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.X_target = []
        self.X_context = []
        self.y = []

        self.data_repr = None
        self.vocab_size = None
        self.subsample_probs = None
        self.model = None
        self.cumulative_table = None
    
    def _build_model(self):

        target_item = Input(shape=(1,), name='target_item')
        context_item = Input(shape=(1,), name='context_item')

        target_embedding_lookup = layers.Embedding(self.vocab_size, self.embedding_size, name='target_embedding', embeddings_initializer = initializers.RandomUniform(seed=kw.RANDOM_STATE))
        context_embedding_lookup = layers.Embedding(self.vocab_size, self.embedding_size, name='context_embedding', embeddings_initializer = initializers.RandomUniform(seed=kw.RANDOM_STATE))

        embedding_target = target_embedding_lookup(target_item)
        embedding_context = context_embedding_lookup(context_item)

        merged_vector = layers.dot([embedding_target, embedding_context], axes=-1)
        reshaped_vector = layers.Reshape((1,))(merged_vector)
        prediction = layers.Activation('sigmoid')(reshaped_vector)

        model = Model(inputs=[target_item, context_item], outputs=prediction)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='binary_crossentropy')

        return model
    
    def timestamp_cum(self, df):

        #Calcula a diferença de tempo entre a iteração atual e a passada, por usuário
        def calc_diff(df_group):
            
            max_range = 1
            min_range = 0.3
            
            df_group = df_group.sort_values(kw.COLUMN_DATETIME)
            scaler = MinMaxScaler(feature_range=(min_range,max_range))
            
            #Calcula a diferença de tempo entre iterações
            df_group[kw.COLUMN_TIME_DIFF]  = df_group[kw.COLUMN_DATETIME] - df_group[kw.COLUMN_DATETIME].shift(1)
            df_group[kw.COLUMN_TIME_DIFF]  = df_group[kw.COLUMN_TIME_DIFF].fillna(pd.to_timedelta(0, unit='s'))
            df_group[kw.COLUMN_TIME_DIFF]  = df_group[kw.COLUMN_TIME_DIFF].astype('int64')/ 10**9
            
            #Trata outliers para que eles não afetem tanto o resultdo final 
            Q1 = df_group[kw.COLUMN_TIME_DIFF].quantile(0.25)
            Q3 = df_group[kw.COLUMN_TIME_DIFF].quantile(0.75)
            IQR = Q3 - Q1
            
            threshold = 3
            df_group[kw.COLUMN_TIME_DIFF] = df_group[kw.COLUMN_TIME_DIFF].clip(upper=(Q3 + threshold * IQR))        
            df_group[kw.COLUMN_TIME_CUMSUM] = df_group[kw.COLUMN_TIME_DIFF].cumsum()
            df_group[kw.COLUMN_TIME_CUMSUM] = scaler.fit_transform(df_group[[kw.COLUMN_TIME_CUMSUM]])
            
            return df_group
            
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
        # Gera a coluna de diferença entre iterações
        df = df.groupby(kw.COLUMN_USER_ID, group_keys=False).apply(calc_diff).reset_index(drop=True)

        return df
    
    def _calculate_subsample_probs(self, item_freq):
        
        total_count = np.sum(item_freq)
        freqs = np.array(item_freq)
        z = freqs / total_count
        self.subsample_probs = (np.sqrt(z/self.subsample_threshold) + 1) * (self.subsample_threshold/z)
        
    def _create_subsample_interactions(self, df, interactions):
        rows, cols = interactions.nonzero()
        mask = np.random.rand(len(cols)) <= self.subsample_probs[cols]
        subsample_rows, subsample_cols, subsample_data = rows[mask], cols[mask], interactions.data[mask]
        subsampled_interactions = csr_matrix((subsample_data, (subsample_rows, subsample_cols)), shape=interactions.shape)
        return subsampled_interactions
    
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
    
    def _negative_examples(self, curr_user, curr_item):

        raw_samps = np.random.rand(self.negative_samples,)
        ss = np.searchsorted(self.cumulative_table, raw_samps)
        pos_mask = (ss == np.take(curr_user, ss, mode='clip'))
        X_context = ss[~pos_mask]

        while len(X_context) < self.negative_samples:
            extra_sample = np.searchsorted(self.cumulative_table, np.random.rand(1,))
            if extra_sample != curr_user[curr_item]:
                X_context = np.concatenate((X_context, extra_sample))

        return X_context
    
    def _calculate_training_samples(self, interaction_list):
    
        result, n_iteractions = 0, 0
        for user_id in range(interaction_list.shape[0]):
            curr_user = interaction_list[user_id].nonzero()[1]
            user_size = curr_user.size
            if (user_size < 2):
                continue
            n_iteractions = n_iteractions + user_size
            result = result + (user_size * (user_size-1))

        return result, n_iteractions

    def _data_generator(self, interaction_list, batch_processing):   
        while True:

            X_target, X_context, y, sample_weights = [], [], [], []

            #Se o processamento em batch não for necessário, retorna os dados de uma vez, mudando apenas os exemplos negativos
            if self.X_target != []:
                for user_id in range(interaction_list.shape[0]):
                    curr_user = interaction_list[user_id].nonzero()[1]
                    user_size = curr_user.size
                    if (user_size < 2):
                        continue
                    X_context.extend(np.tile(curr_user, user_size)[np.tile(np.arange(1, user_size+1), user_size-1) + np.repeat(np.arange(user_size-1)*(user_size+1), user_size)])
                    neg_X_context = []
                    for curr_item in range(user_size):
                        neg_X_context.extend(self._negative_examples(curr_user, curr_item))
                    X_context.extend(neg_X_context)

                yield (np.array(self.X_target), np.array(X_context)), np.array(self.y), np.array(self.sample_weights)
                continue

            for user_id in range(interaction_list.shape[0]):

                #Recebe os usuários da matriz de iteração um a um
                curr_user = interaction_list[user_id].nonzero()[1]
                weights = interaction_list[user_id, curr_user].data
                user_size = curr_user.size

                if (user_size < 2):
                    continue
                    
                # Amostras positivas
                # Cria todas as combinações possíveis de pares de itens para o usuário atual
                user_repeat = np.repeat(range(user_size), user_size-1)
                user_comb = np.tile(range(user_size), user_size)[np.tile(np.arange(1, user_size+1), user_size-1) + np.repeat(np.arange(user_size-1)*(user_size+1), user_size)]  

                X_target.extend(curr_user[user_repeat])
                X_context.extend(curr_user[user_comb])
                y.extend(np.ones(user_size * (user_size-1)))
                sample_weights.extend(np.round(1 - abs(weights[user_repeat] - weights[user_comb]), 2))

                neg_X_context = []
                #Para cada treinamento positivo, retorna N negativos
                for curr_item in range(user_size): 
                    neg_X_context.extend(self._negative_examples(curr_user, curr_item))
                    X_target.extend(np.repeat(curr_user[curr_item], self.negative_samples))

                X_context.extend(neg_X_context)
                y.extend(np.zeros(len(neg_X_context)))
                sample_weights.extend(np.ones(len(neg_X_context)))
                                                                
                #Treina o modelo em batch
                if batch_processing == True:
                    num_batches = int(len(X_target) / self.batch_size)
                    if num_batches > 0:
                        for i in range(0, num_batches * self.batch_size, self.batch_size):
                            yield (np.array(X_target[i:i + self.batch_size]), np.array(X_context[i:i + self.batch_size])), np.array(y[i:i + self.batch_size]), np.array(sample_weights[i:i + self.batch_size])
                        X_target = X_target[num_batches * self.batch_size:]
                        X_context = X_context[num_batches * self.batch_size:]
                        y = y[num_batches * self.batch_size:]

            if batch_processing == False:
                self.X_target = X_target
                self.X_context = X_context
                self.y = y
                self.sample_weights = sample_weights

            yield (np.array(X_target), np.array(X_context)), np.array(y), np.array(sample_weights)
                
    def _save_embeddings(self):
        os.makedirs(self.embedding_dir, exist_ok=True)
        item_embeddings = self.model.get_layer('target_embedding').get_weights()[0]
        np.save(os.path.join(self.embedding_dir, kw.FILE_ITEMS_EMBEDDINGS), item_embeddings)
        pickle.dump(self.data_repr, open(os.path.join(self.embedding_dir, kw.FILE_SPARSE_REPR), 'wb'))

    def fit(self, df):
        
        if os.path.exists(os.path.join(self.embedding_dir, kw.FILE_ITEMS_EMBEDDINGS)):
            return

        np.random.seed(kw.RANDOM_STATE)
        tf.random.set_seed(kw.RANDOM_STATE)

        if kw.COLUMN_TIMESTAMP in df.columns or kw.COLUMN_DATETIME in df.columns:
            df = self.timestamp_cum(df)
        else:
            raise Exception("Timestamp column not found")
                
        # Cria a representacao dos dados a partir do dataset
        self.data_repr = DataRepr(df)
        self.vocab_size = len(self.data_repr.le_items.classes_)
        interaction_list = self.data_repr.get_user_items_matrix()

        # Calcula a probabilidade de descarte de cada item
        item_counts = np.diff(interaction_list.tocsc().indptr)
        self._calculate_subsample_probs(item_counts)
        subsample_interactions = self._create_subsample_interactions(df, interaction_list)

        #Cria a tabela cumulativa
        self.cumulative_table = self._cumulative_table(item_counts)
        
        #Calcula o numero de passos para que o epoch atual termine de rodar
        n_samples, n_interactions = self._calculate_training_samples(subsample_interactions)
        #Adiciona os passos negativos
        self.negative_samples_size = (n_interactions * self.negative_samples)
        n_samples = n_samples + self.negative_samples_size
        steps_per_epoch = np.ceil(n_samples/self.batch_size)

        #Caso a quantidade de amostras seja menor que o batch_size, não é necessário processamento em batch
        batch_processing = steps_per_epoch != 1
                
        #Cria o modelo e inicia o treinamento
        self.model = self._build_model()
        memory_printing_callback = MemoryPrintingCallback()
        
        self.model.fit(self._data_generator(subsample_interactions, batch_processing), 
                  steps_per_epoch=steps_per_epoch, 
                  epochs=self.epochs, 
                  shuffle=False, 
                  verbose=2)
        
        #, callbacks=[memory_printing_callback]
                        
        self._save_embeddings()
        
    def get_embeddings(self):
        embedding_layer = self.model.get_layer('target_embedding')
        return embedding_layer.get_weights()[0]
    
    def get_datarepr(self):
        return self.data_repr