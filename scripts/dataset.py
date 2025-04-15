import os
import pandas as pd
import scripts as kw
import numpy as np

DATASETS_TABLE = pd.DataFrame(
    [[1,  'Anime Recommendations',     'E',         1.0],
     [2,  'BestBuy',                   'I',         1.0],
     [3,  'Book-Crossing',             'E',         1.0],
     [4,  'CiaoDVD',                   'I',         1.0],
     [5,  'DeliciousBookmarks',        'I',         1.0],
     [6,  'Filmtrust',                 'E',         1.0],
     [7,  'Jester',                    'E',         1.0],
     [8,  'Last.FM',                   'I',         1.0],
     [9,  'Last.FM - Tagged',          'I',         1.0],
     [10, 'LibimSeTi',                 'E',         1.0],
     [11, 'MovieLens',                 'E',         1.0],
     [12, 'NetflixPrize',              'E',         0.05],
     [13, 'RetailRocket-All',          'I',         1.0],
     [14, 'RetailRocket-Transactions', 'I',         1.0],
     [15, 'Taobao',                    'I',         1.0],
     [16, 'Events',                    'I',         1.0],
     [17, 'AmazonBooks',               'E',         1.0],
     [18, 'AmazonBeauty',              'E',         1.0],
     [19, 'kuairand-1k',               'E',         1.0],], 
    columns=[kw.DATASET_ID, kw.DATASET_NAME, kw.DATASET_TYPE, kw.DATASET_SAMPLING_RATE]
).set_index(kw.DATASET_ID)


class Dataset(object):

    def __init__(self, id, path):
        self.name = DATASETS_TABLE.loc[id, kw.DATASET_NAME]
        self.sampling_rate = DATASETS_TABLE.loc[id, kw.DATASET_SAMPLING_RATE]
        self.df = pd.read_csv(path, delimiter=kw.DELIMITER, encoding=kw.ENCODING, quoting=kw.QUOTING, quotechar=kw.QUOTECHAR, header=0)
        self.df = self.df.dropna().drop_duplicates(subset=[kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID], keep='last')

        if kw.COLUMN_RATING in self.df.columns:
            explicit_ratings = self.df[kw.COLUMN_RATING]!=-1
            min_max = self.df[explicit_ratings][kw.COLUMN_RATING].apply(['min', 'max'])
            mean_rating = min_max.loc['min'] + (min_max.loc['max']-min_max.loc['min'])/2  
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
def get_datasets(dataset_folder='datasets', datasets=None):
    for dataset_id, dataset_data in DATASETS_TABLE.iterrows():
        if datasets is None or dataset_data[kw.DATASET_NAME] in datasets:
            dataset_filepath = os.path.join(dataset_folder, dataset_data[kw.DATASET_NAME], kw.FILE_INTERACTIONS)
            yield Dataset(dataset_id, dataset_filepath)