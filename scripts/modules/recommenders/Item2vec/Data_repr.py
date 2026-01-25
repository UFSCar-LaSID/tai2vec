import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
import scripts as kw

class DataRepr(object):
    
    def __init__(self, df, temporal_sorting=False):
        self.temporal_sorting = temporal_sorting
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

        if self.temporal_sorting:

            if kw.COLUMN_DATETIME in df.columns:
                df = df.sort_values(by=[kw.COLUMN_USER_ID, kw.COLUMN_DATETIME])
            else:
                df = df.sort_values(by=[kw.COLUMN_USER_ID, kw.COLUMN_TIMESTAMP])

        grouped_items = df.groupby(kw.COLUMN_USER_ID)[kw.COLUMN_ITEM_ID].agg(list)
        return grouped_items.tolist()
    
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