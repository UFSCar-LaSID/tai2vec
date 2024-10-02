import implicit
import numpy as np
import os
import pandas as pd
import pickle 
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

import scripts as kw

class SparseRepr(object):
    
    def __init__(self, df):
        self.le_users = LabelEncoder()
        self.le_users.fit(df[kw.COLUMN_USER_ID])
        self.le_items = LabelEncoder()
        self.le_items.fit(df[kw.COLUMN_ITEM_ID])

    def get_user_items_matrix(self, df):
        data = np.ones(len(df))
        user_ind = self.le_users.transform(df[kw.COLUMN_USER_ID])
        item_ind = self.le_items.transform(df[kw.COLUMN_ITEM_ID])
        n_users = len(self.le_users.classes_)
        n_items = len(self.le_items.classes_)
        return csr_matrix((data, (user_ind, item_ind)), shape=(n_users, n_items))
    
    def get_user_index(self, user_id):
        return self.le_users.transform(user_id)
    
    def get_item_index(self, item_id):
        return self.le_items.transform(item_id)
    
    def get_user_id(self, user_index):
        return self.le_users.inverse_transform(user_index)
    
    def get_item_id(self, item_index):
        return self.le_items.inverse_transform(item_index)


class ImplicitRecommender(object):

    def _timestamp_diff(self, df):

        #Calcula a diferença de tempo entre a iteração atual e a passada, por usuário
        def calc_diff(df_group):
            df_group = df_group.sort_values(kw.COLUMN_DATETIME)
            df_group[kw.COLUMN_TIME_DIFF] = df_group[kw.COLUMN_DATETIME] - df_group[kw.COLUMN_DATETIME].shift(1)
            df_group[kw.COLUMN_TIME_DIFF] = df_group[kw.COLUMN_TIME_DIFF].fillna(pd.to_timedelta(0, unit='s'))
            df_group[kw.COLUMN_TIME_DIFF] = df_group[kw.COLUMN_TIME_DIFF].astype('int64')/ 10**9

            df_group['Q1'] = df_group[kw.COLUMN_TIME_DIFF].quantile(0.25)
            df_group['Q3'] = df_group[kw.COLUMN_TIME_DIFF].quantile(0.75)

            return df_group
        
        if kw.COLUMN_TIMESTAMP in df.columns:
            df[kw.COLUMN_DATETIME] = pd.to_datetime(df[kw.COLUMN_TIMESTAMP], unit='s')
        elif kw.COLUMN_DATETIME in df.columns:
            df[kw.COLUMN_DATETIME] = pd.to_datetime(df[kw.COLUMN_DATETIME])

        # Gera a coluna de diferença entre iterações
        df = df.groupby(kw.COLUMN_USER_ID, group_keys=False).apply(calc_diff).reset_index(drop=True)
        df[kw.COLUMN_THRESHOLD] = df['Q3'] + (2 * (df['Q3'] - df['Q1']))
        
        df['mask'] = df[kw.COLUMN_TIME_DIFF] > df[kw.COLUMN_THRESHOLD]
        df['increment'] = df.groupby(kw.COLUMN_USER_ID)['mask'].cumsum()
        df['old_user_id'] = df[kw.COLUMN_USER_ID]
        df[kw.COLUMN_USER_ID] = df.groupby([kw.COLUMN_USER_ID, 'increment']).ngroup()

        df.drop(columns=['mask', 'increment'], inplace=True)

        return df

    def _save_embeddings(self):
        item_embeddings = self.model.item_factors if kw.TRAIN_MODE == 'cpu' else self.model.item_factors.to_numpy()
        user_embeddings = self.model.user_factors if kw.TRAIN_MODE == 'cpu' else self.model.user_factors.to_numpy()
        np.save(os.path.join(self.embeddings_filepath, kw.FILE_ITEMS_EMBEDDINGS), item_embeddings)
        np.save(os.path.join(self.embeddings_filepath, kw.FILE_USERS_EMBEDDINGS), user_embeddings)
        pickle.dump(self.sparse_repr, open(os.path.join(self.embeddings_filepath, kw.FILE_SPARSE_REPR), 'wb'))


    def fit(self, df_train):
        df_train = self._timestamp_diff(df_train)
        self.sparse_repr = SparseRepr(df_train)
        self.train_user_items_matrix = self.sparse_repr.get_user_items_matrix(df_train)
        self.model.fit(self.train_user_items_matrix, show_progress=False)
        if not os.listdir(self.embeddings_filepath):
            self._save_embeddings()

    def recommend(self, df_test):
        userid = df_test[kw.COLUMN_USER_ID].unique()
        users_sparse_index = self.sparse_repr.get_user_index(userid)        
        recommended_items, _ = self.model.recommend(
            userid=users_sparse_index, 
            user_items=self.train_user_items_matrix[users_sparse_index, :], 
            N=kw.TOP_N,
            filter_already_liked_items=True
        )
        recommendations = pd.DataFrame(
            np.vstack([
                np.repeat(userid, kw.TOP_N), 
                self.sparse_repr.get_item_id(np.ravel(recommended_items))]  # Para que os itens sejam id's e não indexes
            ).T,
            columns=[kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID]
        )
        return recommendations
    

class ALS_time_model(ImplicitRecommender):
    def __init__(self, embeddings_filepath, factors=64, regularization=0.01, iterations=15):
        self.embeddings_filepath = embeddings_filepath        
        ALSModel = implicit.cpu.als.AlternatingLeastSquares if kw.TRAIN_MODE == 'cpu' else implicit.gpu.als.AlternatingLeastSquares
        self.model = ALSModel(factors=factors, regularization=regularization, iterations=iterations, random_state=kw.RANDOM_STATE)

class BPR_time_model(ImplicitRecommender):
    def __init__(self, embeddings_filepath, factors=100, learning_rate=0.01, regularization=0.01, iterations=100):
        self.embeddings_filepath = embeddings_filepath
        BPRModel = implicit.cpu.bpr.BayesianPersonalizedRanking if kw.TRAIN_MODE == 'cpu' else implicit.gpu.bpr.BayesianPersonalizedRanking        
        self.model = BPRModel(factors=factors, learning_rate=learning_rate, regularization=regularization, iterations=iterations, random_state=kw.RANDOM_STATE)