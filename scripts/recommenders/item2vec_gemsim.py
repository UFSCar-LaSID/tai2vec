import implicit
import numpy as np
import os
import pandas as pd
import pickle 
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import scripts as kw

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

class DataRepr(object):
    
    def __init__(self, df):
        self.le_users = LabelEncoder()
        self.le_users.fit(df[kw.COLUMN_USER_ID])
        self.le_items = LabelEncoder()
        self.le_items.fit(df[kw.COLUMN_ITEM_ID])
        self.interaction_matrix = self.create_user_items_matrix(df)
    
    def create_user_items_matrix(self, df):
        data = np.ones(len(df))
        user_ind = self.le_users.transform(df[kw.COLUMN_USER_ID])
        item_ind = self.le_items.transform(df[kw.COLUMN_ITEM_ID])
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

        users, user_pos = np.unique(user_indices, return_index=True)
        interaction_list = np.split(item_indices, user_pos[1:])
        interaction_list = [list(items) for items in interaction_list]
        
        return interaction_list
    
class Word2Vec_gemsim:
    def __init__(self, embeddings_filepath, embedding_size=100, sample=0.0001, negative_sampling=5, ns_exp=0.75, epochs=5, learning_rate=0.25):
        
        self.embedding_size = embedding_size
        self.negative_sampling = negative_sampling
        self.epochs = epochs
        self.ns_exp = ns_exp
        self.sample = sample
        self.embeddings_filepath = embeddings_filepath
        self.batch_size = kw.MEM_SIZE_LIMIT
        self.learning_rate = learning_rate

    def save_embeddings(self):
        os.makedirs(self.embeddings_filepath, exist_ok=True)
        np.save(os.path.join(self.embeddings_filepath, kw.FILE_ITEMS_EMBEDDINGS), self.model.wv.vectors)
        pickle.dump(self.data_repr, open(os.path.join(self.embeddings_filepath, kw.FILE_SPARSE_REPR), 'wb'))

    def fit(self, df):

        self.data_repr = DataRepr(df)
        sparse_matrix = self.data_repr.get_user_items_matrix()
        
        interactions_file = 'item2vec_interactions.temp'    
        with open(interactions_file, 'w') as f:
            for user in range(sparse_matrix.shape[0]):
                f.write(' '.join(sparse_matrix[user].nonzero()[1].astype(str)) + '\n')
                
        cores = os.cpu_count()

        self.model = Word2Vec(
            sentences=LineSentence(interactions_file),
            vector_size=self.embedding_size,
            window=5,
            min_count=1,
            workers=cores-1,
            alpha=self.learning_rate,
            sg=1,
            hs=0,
            negative=self.negative_sampling,
            ns_exponent=self.ns_exp,
            sample=self.sample,
            max_vocab_size=None,
            max_final_vocab=None,
            epochs=self.epochs,
            trim_rule=None,
            sorted_vocab=0,
            batch_words=self.batch_size,
            compute_loss=False,
        )
        
        self.save_embeddings()
        #embeddings = self.model.wv.vectors[np.argsort(np.fromiter(self.model.wv.index_to_key, dtype=np.int32, count=len(self.model.wv.index_to_key)))]