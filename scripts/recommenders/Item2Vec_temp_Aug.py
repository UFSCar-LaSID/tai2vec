import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

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
from keras.optimizers import Adam # type: ignore


class DataRepr(object):
    
    def __init__(self, df):
        self.le_users = LabelEncoder()
        self.le_users.fit(df[kw.COLUMN_USER_ID])
        self.le_items = LabelEncoder()
        self.le_items.fit(df[kw.COLUMN_ITEM_ID])
    
    def create_user_items_matrix(self, df):
        data = np.ones(len(df))
        user_ind = self.le_users.transform(df[kw.COLUMN_USER_ID])
        item_ind = self.le_items.transform(df[kw.COLUMN_ITEM_ID])
        n_users = len(self.le_users.classes_)
        n_items = len(self.le_items.classes_)
        return csr_matrix((data, (user_ind, item_ind)), shape=(n_users, n_items))
    
    def create_interaction_list(self, df):

        df[kw.COLUMN_USER_ID] = self.le_users.transform(df[kw.COLUMN_USER_ID])
        df[kw.COLUMN_ITEM_ID] = self.le_items.transform(df[kw.COLUMN_ITEM_ID])

        if kw.COLUMN_DATETIME in df.columns:
            sorted_df = df.sort_values(by=[kw.COLUMN_USER_ID, kw.COLUMN_DATETIME])
        else:
            sorted_df = df.sort_values(by=[kw.COLUMN_USER_ID, kw.COLUMN_TIMESTAMP])

        grouped_items = sorted_df.groupby(kw.COLUMN_USER_ID)[kw.COLUMN_ITEM_ID].agg(list)
        return grouped_items.tolist()
    
    def get_n_user(self):
        return len(self.le_users.classes_)
    
    def get_n_items(self):
        return len(self.le_items.classes_)
    
    def get_user_index(self, user_id):
        return self.le_users.transform([user_id])[0]
    
    def get_item_index(self, item_id):
        return self.le_items.transform([item_id])[0]
    
    def get_item_id(self, item_index):
        return self.le_items.inverse_transform(np.array(item_index))
    
    def get_item_id(self, item_index):
        return self.le_items.inverse_transform(np.array(item_index)) 
    
    def get_user_items_matrix(self):
        return self.interaction_matrix
    

class MemoryPrintingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
      gpu_dict = tf.config.experimental.get_memory_info('GPU:0')
      tf.print('\n GPU memory details [current: {} gb, peak: {} gb]'.format(
          float(gpu_dict['current']) / (1024 ** 3), 
          float(gpu_dict['peak']) / (1024 ** 3)))
      
class Item2vec_temp_aug_model:
    def __init__(self, embedding_dir, factors=100, w_size=-1, learning_rate=0.25, min_learning_rate = 0.0025, subsample = 0.0001, batch_size = kw.MEM_SIZE_LIMIT, negative_samples=5, negative_exp=0.75, epochs=160, time_exp=1, min_time_diff=300, lr_decay=0.5):
        
        self.embedding_dir = embedding_dir
        self.embedding_size = factors
        self.window_size = w_size
        self.subsample_threshold = subsample
        self.negative_samples = negative_samples
        self.negative_expoent = negative_exp
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate 
        self.epochs = epochs
        self.batch_size = batch_size
        self.time_exp = time_exp
        self.min_time_diff = min_time_diff
        self.lr_decay = lr_decay

        self.X_target = []
        self.X_context = []
        self.y = []

        self.data_repr = None
        self.vocab_size = None
        self.subsample_probs = None
        self.model = None
        self.cumulative_table = None

    class Item2Vec(tf.keras.Model):
        def __init__(self, embedding_size, vocab_size):
            super(Item2vec_temp_aug_model.Item2Vec, self).__init__()
            init_width = 0.5 / embedding_size
            initializer = initializers.RandomUniform(minval=-init_width, maxval=init_width, seed=kw.RANDOM_STATE)
            self.target_embedding = layers.Embedding(vocab_size, embedding_size, name='target_embedding', embeddings_initializer=initializer)
            self.context_embedding = layers.Embedding(vocab_size, embedding_size, name='context_embedding', embeddings_initializer=initializer)
    
        def call(self, inputs):

            # target:  (batch_size, 1)
            # context: (batch_size, negative_sampling+1)
            target_item, context_item = inputs

            if len(target_item.shape) == 2:
                target_item = tf.squeeze(target_item, axis=1)

            # target_embedding:  (batch_size, embedding_size)
            # context_embedding: (batch_size, negative_sampling+1, embedding_size)
            target_embedding = self.target_embedding(target_item)
            context_embedding = self.context_embedding(context_item)

            # dots: (batch_size, negative_sampling+1)
            dots = tf.einsum('be,bce->bc', target_embedding, context_embedding)
            return dots
    
    class SaveEmbeddingsCallback(tf.keras.callbacks.Callback):
        def __init__(self, outer, save_interval=50):
            super(Item2vec_temp_aug_model.SaveEmbeddingsCallback, self).__init__()
            self.outer = outer 
            self.save_interval = save_interval

        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % self.save_interval == 0:
                self.outer._save_embeddings(epoch+1)

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
    
    def _negative_examples(self, curr_user, negative_pairs):

        raw_samps = np.random.rand(negative_pairs,)
        ss = np.searchsorted(self.cumulative_table, raw_samps)
        pos_mask = (ss == np.take(curr_user, ss, mode='clip'))
        X_context = ss[~pos_mask]

        while len(X_context) < (negative_pairs):
            random = np.searchsorted(self.cumulative_table, np.random.rand(1,))
            if random not in curr_user:
                X_context = np.concatenate((X_context, random))

        return X_context

    def _generate_positive_data(self):

        X_target, X_context, sample_weights = [], [], []

        arr = np.arange(len(self.interaction_list))
        np.random.shuffle(arr)

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

        return np.array(X_target), np.array(X_context), np.array(sample_weights)
            
    @tf.function
    def _generate_batches(self, target_items, positive_contexts, weights):

        batch_size = tf.shape(target_items)[0]
        
        #Seleciona os itens negativos
        random_samples = tf.random.uniform(shape=(batch_size * self.negative_samples,), minval=0.0, maxval=1.0, dtype=tf.float32)
        negative_contexts = tf.searchsorted(self.cumulative_table, random_samples, side='right')
        negative_contexts = tf.reshape(negative_contexts, (batch_size, self.negative_samples))
        
        # Concatena o item positivo com o vetor de negativos
        positive_contexts = tf.expand_dims(positive_contexts, axis=1) 
        all_contexts = tf.concat([positive_contexts, negative_contexts], axis=1)
        
        # Define y = 1 para o item positivo e y = 0 para os negativos
        positive_labels = tf.ones((batch_size, 1), dtype=tf.float32)
        negative_labels = tf.zeros((batch_size, self.negative_samples), dtype=tf.float32)
        all_labels = tf.concat([positive_labels, negative_labels], axis=1)
        
        return (target_items, all_contexts), all_labels, weights
    
    def _data_generator(self, X_target, X_context_pos, sample_weights):

        dataset = tf.data.Dataset.from_tensor_slices((X_target, X_context_pos, sample_weights))
        dataset = dataset.batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self._generate_batches, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
                
    def _save_embeddings(self, epoch):
        
        path_components = os.path.normpath(self.embedding_dir).split(os.sep)

        if len(path_components) > 2 and path_components[2] == 'validation':
            embedding_dir = self.embedding_dir + "@epochs={}".format(epoch)
        else:
            embedding_dir = self.embedding_dir

        os.makedirs(embedding_dir, exist_ok=True)
        item_embeddings = self.model.get_layer('target_embedding').get_weights()[0]
        np.save(os.path.join(embedding_dir, kw.FILE_ITEMS_EMBEDDINGS), item_embeddings)
        pickle.dump(self.data_repr, open(os.path.join(embedding_dir, kw.FILE_SPARSE_REPR), 'wb'))

    def fit(self, df):

        epochs_string = "@epochs={}".format(self.epochs)
        if os.path.exists(os.path.join(self.embedding_dir + epochs_string, kw.FILE_ITEMS_EMBEDDINGS)):
            return
        
        #Define os callbacks
        memory_printing_callback = MemoryPrintingCallback()
        epoch_callback = self.SaveEmbeddingsCallback(outer=self, save_interval=40)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=self.lr_decay, patience=3, min_lr=self.min_learning_rate, cooldown=5, verbose=1)
        
        np.random.seed(kw.RANDOM_STATE)
        tf.random.set_seed(kw.RANDOM_STATE)

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
                
        #Cria o modelo e inicia o treinamento
        self.model = self.Item2Vec(self.embedding_size, self.vocab_size)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

        #Formato da saida -> ((batch_size,), (batch_size, negative_samples + 1)), (batch_size, negative_samples + 1)
        X_target, X_context_pos, sample_weights = self._generate_positive_data()
        data = self._data_generator(X_target, X_context_pos, sample_weights)

        self.model.fit(
            data, 
            epochs=self.epochs,
            shuffle=False,
            verbose=1, 
            callbacks=[epoch_callback],
        )
        
    def get_embeddings(self):
        embedding_layer = self.model.get_layer('target_embedding')
        return embedding_layer.get_weights()[0]
    
    def get_datarepr(self):
        return self.data_repr

