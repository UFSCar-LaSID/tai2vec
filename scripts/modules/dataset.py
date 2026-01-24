
import os
import pandas as pd
import scripts as kw
import numpy as np

from scripts.modules.preprocess.amazon_beauty import preprocess_amazon_beauty

DATASETS_TABLE = pd.DataFrame(
    [[1,  'amazon-beauty',             'E',         1.0,    preprocess_amazon_beauty],
     [2,  'amazon-books',              'E',         1.0,    lambda _,__: None],
     [3,  'bestbuy',                   'I',         1.0,    lambda _,__: None],
     [4,  'ciaodvd',                   'I',         1.0,    lambda _,__: None],
     [5,  'ml-100k',                   'E',         1.0,    lambda _,__: None],
     [6,  'ml-1m',                     'E',         1.0,    lambda _,__: None],], 
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