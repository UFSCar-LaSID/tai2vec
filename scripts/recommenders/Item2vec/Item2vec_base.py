import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#os.environ['TF_DETERMINISTIC_OPS'] = '1'

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

import keras
from keras import layers, Model, Input, regularizers, initializers, callbacks, optimizers
from scripts.recommenders.Item2vec.Data_repr import DataRepr
from scripts.recommenders.Item2vec.Item2Vec_abc import Item2vec_abstract
from keras.optimizers import Adam # type: ignore
from keras.optimizers import schedules
      
class Item2vec_model(Item2vec_abstract):
    def __init__(self, embedding_dir, factors=100, w_size=-1, learning_rate=0.25, min_learning_rate = 0.000025 ,subsample = 0.001, batch_size = kw.MEM_SIZE_LIMIT, negative_samples=3, negative_exp=0.75, epochs=100, lr_decay=0.95, regularization=-1):
        super().__init__(embedding_dir, factors, w_size, learning_rate, min_learning_rate, subsample, batch_size, negative_samples, negative_exp, epochs, lr_decay, regularization)

    def _generate_positive_data(self):

        X_target, X_context = [], []

        arr = np.arange(len(self.interaction_list))
        #np.random.shuffle(arr)

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
    
    def _generate_batches(self, target_items, positive_contexts):

        batch_size = tf.shape(target_items)[0]

        # Repete cada target_item para cada contexto (positivo + negativos)
        target_items_repeated = tf.repeat(target_items, self.negative_samples + 1)
        
        # Seleciona os itens negativos
        random_samples = self.tf_generator.uniform(shape=(batch_size * self.negative_samples,), minval=0.0, maxval=1.0, dtype=tf.float32)
        negative_contexts = tf.searchsorted(self.cumulative_table, random_samples, side='right')
        negative_contexts = tf.reshape(negative_contexts, (batch_size, self.negative_samples))
        
        # Concatena o item positivo com o vetor de negativos
        positive_contexts = tf.expand_dims(positive_contexts, axis=1) 
        all_contexts = tf.concat([positive_contexts, negative_contexts], axis=1)
        
        # Define y = 1 para o item positivo e y = 0 para os negativos
        positive_labels = tf.ones((batch_size, 1), dtype=tf.float32)
        negative_labels = tf.zeros((batch_size, self.negative_samples), dtype=tf.float32)
        all_labels = tf.concat([positive_labels, negative_labels], axis=1)

        # Achata os contextos para corresponder aos target_items_repeated
        all_contexts_flat = tf.reshape(all_contexts, [-1])
        all_labels_flat = tf.reshape(all_labels, [-1])
        
        return (target_items_repeated, all_contexts_flat), all_labels_flat


    def _data_generator(self):

        dataset = tf.data.Dataset.from_tensor_slices(self._generate_positive_data())
        #dataset = dataset.shuffle(buffer_size=self.batch_size * 10, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self._generate_batches, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        dataset = dataset.cache()
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def fit(self, df):

        epochs_string = "@epochs={}".format(self.epochs)
        if os.path.exists(os.path.join(self.embedding_dir + epochs_string, kw.FILE_ITEMS_EMBEDDINGS)):
            return
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/" + self.embedding_dir)
        epoch_callback = self.SaveEmbeddingsCallback(outer=self, save_interval=20)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=self.lr_decay, patience=3, min_lr=self.min_learning_rate, cooldown=5, verbose=1)
        
        np.random.seed(kw.RANDOM_STATE)
        tf.random.set_seed(kw.RANDOM_STATE)
        self.tf_generator = tf.random.Generator.from_seed(kw.RANDOM_STATE)

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
                
        self.model = self._build_model()

        printc = self.PrintContextCallback(data)
        lr_decay = callbacks.ReduceLROnPlateau(monitor='loss', factor=self.lr_decay, patience=3, min_lr=self.min_learning_rate, cooldown=5, verbose=1)

        self.model.fit(
            data, 
            epochs=self.epochs,
            shuffle=False,
            verbose=2, 
            callbacks=[epoch_callback],
        )