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
import time

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

        t0_hist = time.time()
        encoded_items = self.data_repr.le_items.transform(df_train[kw.COLUMN_ITEM_ID].to_numpy())
        tmp = pd.DataFrame({
            kw.COLUMN_USER_ID: df_train[kw.COLUMN_USER_ID].to_numpy(),
            '_item_idx': encoded_items
        })

        grouped = tmp.groupby(kw.COLUMN_USER_ID)['_item_idx']
        self.user_histories = {
            user_id: grp.to_numpy(dtype=np.int64, copy=False)
            for user_id, grp in grouped
        }
        print(f"Built {len(self.user_histories):,} user histories in {time.time() - t0_hist:.2f}s")

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
            print(f"Loaded embeddings: {self.item_embeddings.shape} from {embeddings_filepath}")

    def predict(self, df_test, show_progress=True):
        scores = []

        test_users = df_test[kw.COLUMN_USER_ID].to_numpy()
        test_item_idx = self.data_repr.le_items.transform(df_test[kw.COLUMN_ITEM_ID].to_numpy())

        for i in range(len(df_test)):

            user_id = test_users[i]
            item_idx = test_item_idx[i]

            target_item_embedding = self.item_embeddings[item_idx]
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

t0_all = time.time()
for dataset in get_datasets(datasets=dataset_names):

    dataset_name = dataset.get_name()
    t0_dataset = time.time()
    print(f"\n=== Dataset: {dataset_name} ===")
    print('Loading dataset {}...'.format(dataset_name))
    
    df = dataset.get_dataframe()

    if kw.COLUMN_DATETIME in df.columns:
        df[kw.COLUMN_DATETIME] = pd.to_datetime(df[kw.COLUMN_DATETIME]).dt.floor('s')
    elif kw.COLUMN_TIMESTAMP in df.columns:
        df[kw.COLUMN_DATETIME] = pd.to_datetime(df[kw.COLUMN_TIMESTAMP], unit='s')
    else:
        raise ValueError('Coluna temporal não encontrada')
        
    # Ordena o dataframe pela coluna de tempo
    df = df.sort_values(by=kw.COLUMN_DATETIME)

    df = remove_single_interactions(df)

    df_train, df_remaining = train_test_split(df, test_size=0.2, shuffle=False)
    _df_val_aux, df_test = train_test_split(df_remaining, test_size=0.5, shuffle=False)

    df_test = remove_cold_start(df_train, df_test)

    user_counts = df_train[kw.COLUMN_USER_ID].value_counts()
    users_with_one_interaction = user_counts[user_counts == 1].shape[0]
    item_counts = df_train[kw.COLUMN_ITEM_ID].value_counts()
    items_that_appear_once = item_counts[item_counts == 1].shape[0]

    print(
        f"Rows: total={len(df):,} | train={len(df_train):,} | test={len(df_test):,} | "
        f"train users w/1 interaction={users_with_one_interaction:,} | items appearing once={items_that_appear_once:,}"
    )

    for recommender_name in recommender_names:

        pbar.set_description(f'{dataset_name} | {recommender_name}')
        t0_run = time.time()
        print(f"\n[{dataset_name} | {recommender_name}] Starting...")
        
        base_path = os.path.join('results', 'embeddings', kw.TEST, dataset_name, recommender_name)
        if not os.path.isdir(base_path):
            print(f"[{dataset_name} | {recommender_name}] SKIP: embeddings folder not found: {base_path}")
            pbar.update(1)
            continue

        run_folders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
        if not run_folders:
            print(f"[{dataset_name} | {recommender_name}] SKIP: no run folder inside: {base_path}")
            pbar.update(1)
            continue

        run_folder = run_folders[0]
        run_path = os.path.join(base_path, run_folder)
        print(f"[{dataset_name} | {recommender_name}] Using run: {run_folder}")

        model = ItemSim(df_train, dataset.min_rating, dataset.max_rating)

        print("Model initialized. Preparing output paths...")

        reg_file_path = os.path.join('results', 'recommendations', kw.TEST, dataset_name, recommender_name, 'regression_results.csv')
        os.makedirs(os.path.dirname(reg_file_path), exist_ok=True)
        reg_metrics_path = os.path.join('results', 'metrics', kw.TEST, dataset_name, recommender_name, 'regression_metrics.csv')
        os.makedirs(os.path.dirname(reg_metrics_path), exist_ok=True)

        print(f"[{dataset_name} | {recommender_name}] Loading embeddings + fitting...")
        model.fit(recommender_name, run_path)

        print(f"[{dataset_name} | {recommender_name}] Predicting on {len(df_test):,} interactions...")
        pred_df = model.predict(df_test, show_progress=True)

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

        print(
            f"[{dataset_name} | {recommender_name}] Done. RMSE={rmse:.6f} MAE={mae:.6f} "
            f"(took {time.time() - t0_run:.1f}s)"
        )

        pbar.update(1)

    print(f"=== Dataset {dataset_name} finished in {time.time() - t0_dataset:.1f}s ===")

pbar.close()
print(f"\nAll done in {time.time() - t0_all:.1f}s")