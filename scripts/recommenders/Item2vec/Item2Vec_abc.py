import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, Model, initializers, regularizers, callbacks, Input
from keras.optimizers import Adam
import scripts as kw
from scripts.recommenders.Item2vec.Data_repr import DataRepr
import abc
import pickle
from keras.mixed_precision import set_global_policy

# Enable XLA and AMP
set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(True)

class Item2vec_abstract(abc.ABC):
    def __init__(self, embedding_dir, factors=100, w_size=-1, learning_rate=0.25, min_learning_rate = 0.000025 ,subsample = 0.001, batch_size = kw.MEM_SIZE_LIMIT, negative_samples=3, negative_exp=0.75, epochs=200, lr_decay=0.95, regularization=-1):
        
        self.embedding_dir = embedding_dir
        self.embedding_size = factors
        self.window_size = w_size
        self.subsample_threshold = subsample
        self.negative_samples = negative_samples
        self.negative_expoent = negative_exp
        self.learning_rate = learning_rate
        self.epochs = 10
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.regularization = regularization
        self.min_learning_rate = min_learning_rate

        self.X_target = []
        self.X_context = []
        self.y = []

        self.data_repr = None
        self.vocab_size = None
        self.subsample_probs = None
        self.model = None
        self.cumulative_table = None

    @abc.abstractmethod
    def fit(self, df):
        pass

    def _build_model(self):

        target_item = Input(shape=(1,), name='target_item')
        context_item = Input(shape=(1,), name='context_item')

        init_width = 0.5 / self.embedding_size
        initializer = initializers.RandomUniform(minval=-init_width, maxval=init_width, seed=kw.RANDOM_STATE)

        if self.regularization != -1:
            target_embedding_lookup = layers.Embedding(self.vocab_size, self.embedding_size, name='target_embedding', embeddings_initializer = initializer, embeddings_regularizer = regularizers.l2(self.regularization))
            context_embedding_lookup = layers.Embedding(self.vocab_size, self.embedding_size, name='context_embedding', embeddings_initializer = initializer, embeddings_regularizer = regularizers.l2(self.regularization))
        else:
            target_embedding_lookup = layers.Embedding(self.vocab_size, self.embedding_size, name='target_embedding', embeddings_initializer = initializer)
            context_embedding_lookup = layers.Embedding(self.vocab_size, self.embedding_size, name='context_embedding', embeddings_initializer = initializer)

        embedding_target = target_embedding_lookup(target_item)
        embedding_context = context_embedding_lookup(context_item)

        merged_vector = layers.dot([embedding_target, embedding_context], axes=-1)
        reshaped_vector = layers.Reshape((1,))(merged_vector)
        prediction = layers.Activation('sigmoid')(reshaped_vector)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = self.learning_rate, 
            decay_steps = self.steps_per_epoch, 
            decay_rate = self.lr_decay, staircase=True)
        
        model = Model(inputs=[target_item, context_item], outputs=prediction)
        model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='binary_crossentropy', jit_compile=True)

        return model
    
    class SaveEmbeddingsCallback(tf.keras.callbacks.Callback):
        def __init__(self, outer, save_interval=20):
            super(Item2vec_abstract.SaveEmbeddingsCallback, self).__init__()
            self.outer = outer 
            self.save_interval = save_interval

        def on_epoch_end(self, epoch, logs=None):
            #if ((epoch + 1) % self.save_interval == 0) or (epoch+1 == 5) or (epoch+1 == 10):
            if (epoch+1 == 5) or (epoch+1 == 10) or (epoch+1 == 20) or (epoch+1 == 30) or (epoch+1 == 40) or (epoch+1 == 50) or (epoch+1 == 75)  or (epoch+1 == 100) or (epoch+1 == 150) or (epoch+1 == 200):
                print("\nSaving embeddings...")
                self.outer._save_embeddings(epoch+1)

    #Define os callbacks
    class PrintContextCallback(tf.keras.callbacks.Callback):
        def __init__(self, dataset):
            super(Item2vec_abstract.PrintContextCallback, self).__init__()
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
    
    def get_embeddings(self):
        embedding_layer = self.model.get_layer('target_embedding')
        return embedding_layer.get_weights()[0]
    
    def get_datarepr(self):
        return self.data_repr