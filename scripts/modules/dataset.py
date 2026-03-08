
import os
import pandas as pd
import scripts as kw
import numpy as np

from scripts.modules.preprocess.amazon_beauty import preprocess_amazon_beauty
from scripts.modules.preprocess.ciaodvd import preprocess_ciaodvd
from scripts.modules.preprocess.ml1m import preprocess_ml1m
from scripts.modules.preprocess.bestbuy import preprocess_bestbuy
from scripts.modules.preprocess.ml100k import preprocess_ml100k
from scripts.modules.preprocess.amazon_books import preprocess_amazon_books

DATASETS_TABLE = pd.DataFrame(
    [[1,  'amazon-beauty',             'E',         1.0,    preprocess_amazon_beauty],
     [2,  'amazon-books',              'E',         1.0,    preprocess_amazon_books],
     [3,  'bestbuy',                   'I',         1.0,    preprocess_bestbuy],
     [4,  'ciaodvd',                   'I',         1.0,    preprocess_ciaodvd],
     [5,  'ml-100k',                   'E',         1.0,    preprocess_ml100k],
     [6,  'ml-1m',                     'E',         1.0,    preprocess_ml1m],], 
    columns=[kw.DATASET_ID, kw.DATASET_NAME, kw.DATASET_TYPE, kw.DATASET_SAMPLING_RATE, kw.DATASET_PREPROCESS_FUNCTION]
).set_index(kw.DATASET_ID)

class Dataset(object):

    def __init__(self, id, path):
        self.name = DATASETS_TABLE.loc[id, kw.DATASET_NAME]
        self.sampling_rate = DATASETS_TABLE.loc[id, kw.DATASET_SAMPLING_RATE]
        self.df = pd.read_csv(path, delimiter=kw.DELIMITER, encoding=kw.ENCODING, quoting=kw.QUOTING, quotechar=kw.QUOTECHAR, header=0)
        self.df = self.df.dropna().drop_duplicates(subset=[kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID], keep='last')

        if kw.COLUMN_RATING in self.df.columns:
            explicit_ratings = self.df[kw.COLUMN_RATING]!=-1
            min_max = self.df[explicit_ratings][kw.COLUMN_RATING].agg(['min', 'max'])
            mean_rating = (min_max.loc['min'] + min_max.loc['max']) / 2  
            self.df = self.df[(self.df[kw.COLUMN_RATING]>=mean_rating)|(self.df[kw.COLUMN_RATING]==-1)]
            self.min_rating = min_max.loc['min']
            self.max_rating = min_max.loc['max']

        if self.sampling_rate < 1.0:
            self.df = self.sample_dataset(self.df)

    def sample_dataset(self, df):
        unique_users = df['id_user'].unique()
        num_users_to_remove = int(len(unique_users) * (1-self.sampling_rate))
        users_to_remove = np.random.choice(unique_users, num_users_to_remove, replace=False)
        return df[~df['id_user'].isin(users_to_remove)]
    
    def get_name(self):
        return self.name
    
    def get_sampling_rate(self):
        return self.sampling_rate

    def get_dataframe(self):
        return self.df

    def get_n_users(self):
        return self.df[kw.COLUMN_USER_ID].nunique()

    def get_n_items(self):
        return self.df[kw.COLUMN_ITEM_ID].nunique()

    def get_n_interactions(self):
        return len(self.df)


# Recupera um conjunto de datasets, retornando um de cada vez
def get_datasets(dataset_folder=kw.DATASET_PATH, datasets=None):
    for dataset_id, dataset_data in DATASETS_TABLE.iterrows():
        if datasets is None or dataset_data[kw.DATASET_NAME] in datasets:
            dataset_filepath = os.path.join(dataset_folder, dataset_data[kw.DATASET_NAME], kw.FILE_INTERACTIONS)
            yield Dataset(dataset_id, dataset_filepath)

def remove_single_interactions(df):
    while True:
        count_users = df[kw.COLUMN_USER_ID].value_counts(sort=False)
        count_items = df[kw.COLUMN_ITEM_ID].value_counts(sort=False)
        invalid_users = count_users[count_users==1].index
        invalid_items = count_items[count_items==1].index
        if len(invalid_users) == 0 and len(invalid_items) == 0:
            break
        df = df[(~df[kw.COLUMN_USER_ID].isin(invalid_users))&(~df[kw.COLUMN_ITEM_ID].isin(invalid_items))].copy()
    return df

def remove_cold_start(df_train, df_test):
    valid_users = df_test[kw.COLUMN_USER_ID].isin(df_train[kw.COLUMN_USER_ID])
    valid_items = df_test[kw.COLUMN_ITEM_ID].isin(df_train[kw.COLUMN_ITEM_ID])
    return df_test[(valid_users)&(valid_items)].copy()