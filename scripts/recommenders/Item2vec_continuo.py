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
    
    def create_column_list(self, df, column, transform = False):

        if column not in df.columns:
            raise Exception("Coluna não encontrada")

        if transform:
            df[kw.COLUMN_USER_ID] = self.le_users.transform(df[kw.COLUMN_USER_ID])
            df[kw.COLUMN_ITEM_ID] = self.le_items.transform(df[kw.COLUMN_ITEM_ID])

        if kw.COLUMN_DATETIME in df.columns:
            sorted_df = df.sort_values(by=[kw.COLUMN_USER_ID, kw.COLUMN_DATETIME])
        else:
            sorted_df = df.sort_values(by=[kw.COLUMN_USER_ID, kw.COLUMN_TIMESTAMP])

        grouped_items = sorted_df.groupby(kw.COLUMN_USER_ID)[column].agg(list)
        return grouped_items.tolist()
    
    def create_metrics_list(self, df, column):
        return df.groupby(kw.COLUMN_USER_ID)[column].first().tolist()
    
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
      
class Item2vec_Temp_Cont_model:
    def __init__(self, embedding_dir, factors=100, w_size=-1, learning_rate=0.25, min_learning_rate = 0.025, subsample = 0.0001, batch_size = kw.MEM_SIZE_LIMIT, negative_samples=5, negative_exp=0.75, min_weight = 0.3, curve_exp = 2, epochs=200, min_time_diff=60, weight_floor=0.3):
        
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
        self.min_weight = min_weight
        self.weight_floor = weight_floor
        self.curve_exp = curve_exp
        self.min_time_diff = min_time_diff

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

        initializer = initializers.RandomUniform(seed=kw.RANDOM_STATE)
        #initializer = initializers.RandomNormal(stddev=0.1, seed=kw.RANDOM_STATE)

        target_embedding_lookup = layers.Embedding(self.vocab_size, self.embedding_size, name='target_embedding', embeddings_initializer = initializer)
        context_embedding_lookup = layers.Embedding(self.vocab_size, self.embedding_size, name='context_embedding', embeddings_initializer = initializer)

        embedding_target = target_embedding_lookup(target_item)
        embedding_context = context_embedding_lookup(context_item)

        merged_vector = layers.dot([embedding_target, embedding_context], axes=-1)
        reshaped_vector = layers.Reshape((1,))(merged_vector)
        prediction = layers.Activation('sigmoid')(reshaped_vector)

        #lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        #    initial_learning_rate = self.learning_rate, decay_steps = self.steps_per_epoch , decay_rate=0.98, staircase=True)

        model = Model(inputs=[target_item, context_item], outputs=prediction)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='binary_crossentropy')

        return model
    
    def timestamp_cum(self, df):

        #Calcula a diferença de tempo entre a iteração atual e a passada, por usuário
        def calc_diff(df_group):
            
            max_range = 1
            min_range = self.min_weight
            
            df_group = df_group.sort_values(kw.COLUMN_DATETIME)
            scaler = MinMaxScaler(feature_range=(min_range,max_range))
            
            #Calcula a diferença de tempo entre iterações
            df_group[kw.COLUMN_TIME_DIFF]  = df_group[kw.COLUMN_DATETIME] - df_group[kw.COLUMN_DATETIME].shift(1)
            df_group[kw.COLUMN_TIME_DIFF]  = df_group[kw.COLUMN_TIME_DIFF].fillna(pd.to_timedelta(0, unit='s'))
            df_group[kw.COLUMN_TIME_DIFF]  = df_group[kw.COLUMN_TIME_DIFF].astype('int64')/ 10**9

            #Desconsidera interações que aconteceram em um pequeno intervalo de tempo
            non_noise_diffs = df_group[df_group['timestamp_diff'] > self.min_time_diff]
            
            #Trata outliers para que eles não afetem tanto o resultdo final 
            Q1 = non_noise_diffs[kw.COLUMN_TIME_DIFF].quantile(0.25)
            Q3 = non_noise_diffs[kw.COLUMN_TIME_DIFF].quantile(0.75)
            IQR = Q3 - Q1
            
            threshold = 1.5
            df_group[kw.COLUMN_TIME_DIFF] = df_group[kw.COLUMN_TIME_DIFF].clip(upper=(Q3 + threshold * IQR))        
            df_group[kw.COLUMN_TIME_CUMSUM] = df_group[kw.COLUMN_TIME_DIFF].cumsum()
            df_group[kw.COLUMN_MEAN] = non_noise_diffs[kw.COLUMN_TIME_DIFF].mean()
            df_group[kw.COLUMN_STD] = non_noise_diffs[kw.COLUMN_TIME_DIFF].std()
            df_group[kw.COLUMN_MEAN] = df_group[kw.COLUMN_MEAN].fillna(0)
            df_group[kw.COLUMN_STD] = df_group[kw.COLUMN_STD].fillna(0)
            df_group[kw.COLUMN_TIME_CUMSUM_NORM] = scaler.fit_transform(df_group[[kw.COLUMN_TIME_CUMSUM]])
            
            return df_group
            
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
        # Gera a coluna de diferença entre iterações
        df = df.groupby(kw.COLUMN_USER_ID, group_keys=False).apply(calc_diff).reset_index(drop=True)

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
    
    def _calculate_all_training_samples(self):

        #Calcula o numero de passos para que o epoch atual termine de rodar
        n_samples, n_interactions = self._calculate_positive_training_samples()
        #Adiciona os passos negativos
        self.negative_samples_size = (n_interactions * self.negative_samples)
        return n_samples + self.negative_samples_size

    def _calculate_positive_training_samples(self):
    
        result, n_iteractions = 0, 0

        for user_id in range(len(self.interaction_list)):

            curr_user = self.interaction_list[user_id]
            user_size = len(curr_user)

            if (user_size < 2):
                continue

            n_iteractions = n_iteractions + user_size

            # Muda o calculo realizado caso a janela de contexto seja infinita
            if self.window_size == -1:
                result = result + (user_size * (user_size-1))
            else:
                for i in range(user_size):
                    start_idx = max(0, i - self.window_size)
                    end_idx = min(user_size, i + self.window_size + 1)
                    num_context_items = end_idx - start_idx
                    result += num_context_items

        return result, n_iteractions

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
    
    def _data_generator(self, batch_processing):   

        while True:

            X_target, X_context, y, sample_weights = [], [], [], []

            #Se o processamento em batch não for necessário, retorna os dados de uma vez, mudando apenas o contexto
            if self.X_target != []:

                for user_id in range(len(self.interaction_list)):
                    curr_user = self.interaction_list[user_id]
                    user_size = len(curr_user)
                    if (user_size < 2):
                        continue
                    X_context.extend(np.tile(curr_user, user_size)[np.tile(np.arange(1, user_size+1), user_size-1) + np.repeat(np.arange(user_size-1)*(user_size+1), user_size)])
                    neg_X_context = []
                    for curr_item in range(user_size):
                        neg_X_context.extend(self._negative_examples(curr_user, curr_item))
                    X_context.extend(neg_X_context)

                yield (np.array(self.X_target), np.array(X_context)), np.array(self.y), np.array(self.sample_weights)
                continue

            arr = np.arange(len(self.interaction_list))
            np.random.shuffle(arr)

            for user_id in arr:

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

                    X_target.extend(curr_user[user_repeat])
                    X_context.extend(curr_user[user_comb])
                    y.extend(np.ones(user_size * (user_size-1)))
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
                        X_target.extend(np.repeat(curr_user[i], len(context_indices)))
                        X_context.extend(np.array(curr_user)[context_indices])
                        y.extend(np.ones(len(context_indices)))
                        norm_sample_weights_aux = 1 - abs(norm_weights[i] - norm_weights[context_indices])
                        norm_sample_weights = np.round(np.maximum(1 - (np.log10(1/norm_sample_weights_aux)), self.weight_floor), 2)
                        if self.curve_exp == -1:
                            sample_weights.extend(norm_sample_weights)
                        else:
                            z_sample_weights = self._calculate_weights(abs(cumulative_time[i] - cumulative_time[context_indices]), mean, std)
                            final_weights = (z_sample_weights + norm_sample_weights)/2
                            sample_weights.extend(final_weights)

                neg_X_context = []
                #Para cada treinamento positivo, retorna N negativos
                for curr_item in range(user_size): 
                    neg_X_context.extend(self._negative_examples(curr_user, curr_item))
                    X_target.extend(np.repeat(curr_user[curr_item], self.negative_samples))

                X_context.extend(neg_X_context)
                y.extend(np.zeros(len(neg_X_context)))
                #sample_weights.extend(np.ones(len(neg_X_context)))

                norm_mean = np.mean(norm_weights)
                sample_weights.extend(np.repeat(norm_mean, len(neg_X_context)))

                #Treina o modelo em batch
                if batch_processing == True:
                    num_batches = int(len(X_target) / self.batch_size)
                    if num_batches > 0:
                        for i in range(0, num_batches * self.batch_size, self.batch_size):
                            yield (np.array(X_target[i:i + self.batch_size]), np.array(X_context[i:i + self.batch_size])), np.array(y[i:i + self.batch_size]), np.array(sample_weights[i:i + self.batch_size])
                        X_target = X_target[num_batches * self.batch_size:]
                        X_context = X_context[num_batches * self.batch_size:]
                        y = y[num_batches * self.batch_size:]
                        sample_weights = sample_weights[num_batches * self.batch_size:]

            if batch_processing == False:
                self.X_target = X_target
                self.X_context = X_context
                self.y = y
                self.sample_weights = sample_weights

            yield (np.array(X_target), np.array(X_context)), np.array(y), np.array(sample_weights)

    class SaveEmbeddingsCallback(tf.keras.callbacks.Callback):
        def __init__(self, outer, save_interval=50):
            super(Item2vec_Temp_Cont_model.SaveEmbeddingsCallback, self).__init__()
            self.outer = outer 
            self.save_interval = save_interval

        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % self.save_interval == 0:
                self.outer._save_embeddings(epoch+1)
                
    def _save_embeddings(self, epoch):
        embedding_dir = self.embedding_dir + "_epochs-{}".format(epoch)
        os.makedirs(embedding_dir, exist_ok=True)
        item_embeddings = self.model.get_layer('target_embedding').get_weights()[0]
        np.save(os.path.join(embedding_dir, kw.FILE_ITEMS_EMBEDDINGS), item_embeddings)
        pickle.dump(self.data_repr, open(os.path.join(embedding_dir, kw.FILE_SPARSE_REPR), 'wb'))

    def fit(self, df):

        epochs_string = "_epochs-{}".format(self.epochs)
        if os.path.exists(os.path.join(self.embedding_dir + epochs_string, kw.FILE_ITEMS_EMBEDDINGS)):
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

        # Reduz o dataset com subsampling e cria a lista de interações e de pesos
        df = self._subsample_items(df)

        # Cria as listas de interações e pesos
        self.interaction_list = self.data_repr.create_column_list(df, kw.COLUMN_ITEM_ID, transform=True)
        self.cumsum_list = self.data_repr.create_column_list(df, kw.COLUMN_TIME_CUMSUM)
        self.norm_weight_list = self.data_repr.create_column_list(df, kw.COLUMN_TIME_CUMSUM_NORM)
        self.mean_list = self.data_repr.create_metrics_list(df, kw.COLUMN_MEAN)
        self.std_list = self.data_repr.create_metrics_list(df, kw.COLUMN_STD)

        #Cria a tabela cumulativa que será utilizada para o negative sampling
        item_freq = df.groupby(kw.COLUMN_ITEM_ID).size().values
        self.cumulative_table = self._cumulative_table(item_freq)

        #Calcula o número de samples, passos por época e se é necessário processamento em batch
        n_samples = self._calculate_all_training_samples()
        self.steps_per_epoch = (n_samples//self.batch_size) + 1
        batch_processing = (self.steps_per_epoch != 1) or (self.window_size != -1)
        batch_processing = True

        print('Number of samples: {}'.format(n_samples))
        print('Batch processing: {}'.format(batch_processing))
                
        #Cria o modelo e inicia o treinamento
        self.model = self._build_model()

        #Define os callbacks
        memory_printing_callback = MemoryPrintingCallback()
        epoch_callback = self.SaveEmbeddingsCallback(outer=self, save_interval=20)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=self.min_learning_rate, cooldown=5, verbose=1)
        
        self.model.fit(self._data_generator(batch_processing), 
                  steps_per_epoch=self.steps_per_epoch, 
                  epochs=self.epochs, 
                  shuffle=False, 
                  verbose=2, callbacks=[epoch_callback, reduce_lr])
     
    def get_embeddings(self):
        embedding_layer = self.model.get_layer('target_embedding')
        return embedding_layer.get_weights()[0]
    
    def get_datarepr(self):
        return self.data_repr