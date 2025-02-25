import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf

import implicit
import scripts as kw

import time

import keras
from keras import layers, Model, Input, regularizers, initializers, callbacks, optimizers
from keras.optimizers import Adam # type: ignore
from keras.optimizers import schedules

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

        df = df.sample(frac = 1)

        grouped_items = df.groupby(kw.COLUMN_USER_ID)[kw.COLUMN_ITEM_ID].agg(list)
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
      
class Item2vec_model:
    def __init__(self, embedding_dir, factors=50, w_size=-1, learning_rate=0.25, min_learning_rate = 0.0025 ,subsample = 0.001, batch_size = kw.MEM_SIZE_LIMIT, negative_samples=3, negative_exp=0.75, epochs=160, lr_decay=0.1, regularization=-1):
        
        self.embedding_dir = embedding_dir
        self.embedding_size = factors
        self.window_size = w_size
        self.subsample_threshold = subsample
        self.negative_samples = negative_samples
        self.negative_expoent = negative_exp
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.min_learning_rate = min_learning_rate
        self.lr_decay = lr_decay
        self.regularization = regularization

        self.X_target = []
        self.X_context = []
        self.y = []

        self.data_repr = None
        self.vocab_size = None
        self.subsample_probs = None
        self.model = None
        self.cumulative_table = None


    class Item2Vec(tf.keras.Model):
        def __init__(self, embedding_size, vocab_size, regularization):
            super(Item2vec_model.Item2Vec, self).__init__()
            init_width = 0.5 / embedding_size
            initializer = initializers.RandomUniform(minval=-init_width, maxval=init_width, seed=kw.RANDOM_STATE)
            if regularization == -1:
                self.target_embedding = layers.Embedding(vocab_size, embedding_size, name='target_embedding', embeddings_initializer=initializer)
                self.context_embedding = layers.Embedding(vocab_size, embedding_size, name='context_embedding', embeddings_initializer=initializer)
            else:
                self.target_embedding = layers.Embedding(vocab_size, embedding_size, name='target_embedding', embeddings_initializer=initializer, embeddings_regularizer=regularizers.l2(regularization))
                self.context_embedding = layers.Embedding(vocab_size, embedding_size, name='context_embedding', embeddings_initializer=initializer, embeddings_regularizer=regularizers.l2(regularization))
    
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
            super(Item2vec_model.SaveEmbeddingsCallback, self).__init__()
            self.outer = outer 
            self.save_interval = save_interval

        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % self.save_interval == 0:
                self.outer._save_embeddings(epoch+1)

    #Define os callbacks
    class PrintContextCallback(tf.keras.callbacks.Callback):
        def __init__(self, dataset):
            super(Item2vec_model.PrintContextCallback, self).__init__()
            self.dataset = dataset

        def on_epoch_end(self, epoch, logs=None):
            for (target, context), y in self.dataset.take(1): 
                print(f"\nEpoch {epoch + 1} - Context Array:")
                print(context.numpy())
                break 
    
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

    def _generate_positive_data(self):

        X_target, X_context = [], []

        arr = np.arange(len(self.interaction_list))
        np.random.shuffle(arr)

        for user_id in arr:

            X_target_aux, X_context_aux = [], []

            #Recebe a lista de itens do usuário atual
            curr_user = self.interaction_list[user_id]
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
                    # Calcula os ids positivos
                    X_target_aux.extend(np.repeat(curr_user[i], len(context_indices)))
                    X_context_aux.extend(np.array(curr_user)[context_indices])

            X_target.extend(X_target_aux)
            X_context.extend(X_context_aux)
            self.steps_per_epoch = (len(X_target) // self.batch_size) + 1

        print("\nNumber of samples:", len(X_target))
        print("Number of negative samples:", len(X_target) * self.negative_samples)

        return np.array(X_target), np.array(X_context)
    
    #@tf.function
    def _generate_batches(self, target_items, positive_contexts):

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
        
        return (target_items, all_contexts), all_labels
        
    def _data_generator(self):

        dataset = tf.data.Dataset.from_tensor_slices(self._generate_positive_data())
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
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/" + self.embedding_dir)
        memory_printing_callback = MemoryPrintingCallback()
        epoch_callback = self.SaveEmbeddingsCallback(outer=self, save_interval=40)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=self.lr_decay, patience=3, min_lr=self.min_learning_rate, cooldown=5, verbose=1)
        
        np.random.seed(kw.RANDOM_STATE)
        #tf.random.set_seed(kw.RANDOM_STATE)

        print(self.regularization)

        # Cria a representacao dos dados a partir do dataset
        self.data_repr = DataRepr(df)
        self.vocab_size = len(self.data_repr.le_items.classes_)

        # Reduz o dataset com subsampling e cria a lista de interações
        df = self._subsample_items(df)
        self.interaction_list = self.data_repr.create_interaction_list(df)

        #Cria a tabela cumulativa que será utilizada para o negative sampling
        self.item_freq = list(df.groupby(kw.COLUMN_ITEM_ID).size().values)
        self.cumulative_table = tf.constant(self._cumulative_table(self.item_freq), dtype=tf.float32)

        #Formato da saida -> ((batch_size,), (batch_size, negative_samples + 1)), (batch_size, negative_samples + 1)
        data = self._data_generator()

        learning_rate_decay_factor = (self.min_learning_rate / self.learning_rate)**(1/self.epochs)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=self.steps_per_epoch,
            decay_rate= learning_rate_decay_factor,
            staircase=True
        )
                
        #Cria o modelo e inicia o treinamento
        self.model = self.Item2Vec(self.embedding_size, self.vocab_size, self.regularization)
        self.model.compile(optimizer= Adam(learning_rate=self.learning_rate),
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

        printc = self.PrintContextCallback(data)

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
    