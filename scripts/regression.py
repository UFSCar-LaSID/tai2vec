import sys
import os

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(parent_path)

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import pandas as pd
import scripts as kw
import numpy as np
from scripts.modules.dataset import get_datasets 
from scripts.modules.utils.parameters_handle import get_input
from scripts.modules.dataset import DATASETS_TABLE, remove_single_interactions, remove_cold_start
from scripts.modules.recommenders import RECOMMENDERS_TABLE
from scripts.modules.recommenders.Item2vec.Data_repr import DataRepr

import torch
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error

def combine_embeddings(target_embeddings, context_embeddings, combination_strategy='avg_norm_after', use_norm=True):

    t = torch.from_numpy(target_embeddings).float()
    c = torch.from_numpy(context_embeddings).float()

    def _norm(x):
        return x / (x.norm(dim=1, keepdim=True) + 1e-9)

    if use_norm:
        t = _norm(t)
        c = _norm(c)
        
    combined = t + c

    return combined.numpy()

class ItemSim:

    def __init__(self, df_train, min_rating, max_rating, use_norm=True, combination_strategy='avg_norm_after'):
        
        self.use_norm = use_norm
        self.combination_strategy = combination_strategy

        self.data_repr = DataRepr(df_train)
        self.user_histories = df_train.groupby(kw.COLUMN_USER_ID)[kw.COLUMN_ITEM_ID].apply(lambda row: self.data_repr.le_items.transform(row.tolist())).to_dict()

        self.min_rating = min_rating
        self.max_rating = max_rating

    def fit(self, algo_name, embeddings_filepath):
        if algo_name == 'ALS' or algo_name == 'BPR':
            self.item_embeddings = np.load(os.path.join(embeddings_filepath, kw.FILE_ITEMS_EMBEDDINGS))
        else:
            target_embeddings = np.load(os.path.join(embeddings_filepath, kw.FILE_ITEMS_EMBEDDINGS))
            context_embeddings = np.load(os.path.join(embeddings_filepath, kw.FILE_CONTEXT_EMBEDDINGS))

            self.item_embeddings = combine_embeddings(
                target_embeddings,
                context_embeddings,
                combination_strategy=self.combination_strategy,
                use_norm=self.use_norm
            )
            print(self.item_embeddings.shape)

    def predict(self, df_test):
        scores = []
        for user_id, item_id in zip(df_test[kw.COLUMN_USER_ID], df_test[kw.COLUMN_ITEM_ID]):
            target_item_embedding = self.item_embeddings[self.data_repr.get_item_index(item_id)]
            user_history_embeddings = self.item_embeddings[self.user_histories[user_id]]
            similarity = cosine_similarity([target_item_embedding], user_history_embeddings).mean()

            score = (similarity + 1) / (2) * (self.max_rating - self.min_rating) + self.min_rating
            
            scores.append(score)
        
        final_df = pd.DataFrame({
            kw.COLUMN_USER_ID: df_test[kw.COLUMN_USER_ID],
            kw.COLUMN_ITEM_ID: df_test[kw.COLUMN_ITEM_ID],
            kw.COLUMN_RATING: df_test[kw.COLUMN_RATING],
            'predicted_rating': scores
        })
        return final_df

dataset_options, recommender_options = get_input('Choose algorithmns and recommenders options to test', [
    {
        'name': 'datasets',
        'description': 'Dataset names (or indexes) to use. If not provided, a interactive menu will be shown. If "all" is provided, all datasets will be preprocessed.',
        'options': DATASETS_TABLE,
        'name_column': kw.DATASET_NAME
    },
    {
        'name': 'recommenders',
        'description': 'Recommender names (or indexes) to use. If not provided, a interactive menu will be shown. If "all" is provided, all recommenders will be used.',
        'options': RECOMMENDERS_TABLE,
        'name_column': kw.RECOMMENDER_NAME
    }
])

dataset_names = [DATASETS_TABLE.loc[option_index, kw.DATASET_NAME] for option_index in dataset_options]
recommender_names = [RECOMMENDERS_TABLE.loc[option_index, kw.RECOMMENDER_NAME] for option_index in recommender_options]

pbar = tqdm(total=len(dataset_names) * len(recommender_names), desc='Processing datasets and recommenders')

for dataset in get_datasets(datasets=dataset_names):

    dataset_name = dataset.get_name()
    print('Loading dataset {}...'.format(dataset_name))
    
    df = dataset.get_dataframe()

    if kw.COLUMN_TIMESTAMP in df.columns:
        df[kw.COLUMN_DATETIME] = pd.to_datetime(df[kw.COLUMN_TIMESTAMP], unit='s')
    elif kw.COLUMN_DATETIME in df.columns:
        df[kw.COLUMN_DATETIME] = pd.to_datetime(df[kw.COLUMN_DATETIME]).dt.floor('s')
    else:
        raise ValueError('Coluna temporal não encontrada')
        
    # Ordena o dataframe pela coluna de tempo
    df = df.sort_values(by=kw.COLUMN_DATETIME)

    df = remove_single_interactions(df)

    df_train, df_remaining = train_test_split(df, test_size=0.3, shuffle=False)
    _df_val_aux, df_test = train_test_split(df_remaining, test_size=0.5, shuffle=False)

    df_test = remove_cold_start(df_train, df_test)

    user_counts = df_train[kw.COLUMN_USER_ID].value_counts()
    users_with_one_interaction = user_counts[user_counts == 1].shape[0]
    item_counts = df_train[kw.COLUMN_ITEM_ID].value_counts()
    items_that_appear_once = item_counts[item_counts == 1].shape[0]

    for recommender_name in recommender_names:

        pbar.set_description(f'{dataset_name} | {recommender_name}')
        
        base_path = os.path.join('results', 'embeddings', kw.TEST, dataset_name, recommender_name)
        run_folder = os.listdir(base_path)[0]
        run_path = os.path.join(base_path, run_folder)

        model = ItemSim(df_train, dataset.min_rating, dataset.max_rating)

        reg_file_path = os.path.join('results', 'recommendations', kw.TEST, dataset_name, recommender_name, 'regression_results.csv')
        os.makedirs(os.path.dirname(reg_file_path), exist_ok=True)
        reg_metrics_path = os.path.join('results', 'metrics', kw.TEST, dataset_name, recommender_name, 'regression_metrics.csv')
        os.makedirs(os.path.dirname(reg_metrics_path), exist_ok=True)

        model.fit(recommender_name, run_path)
        pred_df = model.predict(df_test)

        rmse = np.sqrt(mean_squared_error(pred_df[kw.COLUMN_RATING], pred_df['predicted_rating']))
        mae = mean_absolute_error(pred_df[kw.COLUMN_RATING], pred_df['predicted_rating'])

        pred_df.to_csv(reg_file_path, index=False)

        metrics_df = pd.DataFrame({
            'dataset': [dataset_name],
            'recommender': [recommender_name],
            'rmse': [rmse],
            'mae': [mae]
        })
        metrics_df.to_csv(reg_metrics_path, index=False)

        pbar.update(1)