import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser

import scripts as kw
from scripts.dataset import get_datasets
from scripts.file_handlers import (
    get_embeddings_filepath,
    get_recomendation_filepath,
    get_metrics_filepath,
    str_to_dict,
    log_recommendations,
)
from scripts.recommenders import get_recommenders
from scripts.recsys import remove_single_interactions, remove_cold_start
from scripts.metrics import Metrics


DATASETS = ['ciaodvd', 'amazon-books', 'amazon-beauty']
RECOMMENDERS = ['Item2Vec_itemSim', 'TimeI2V_Disc_Aug', 'TimeI2V_Cont']
BEST_COLUMN = kw.EVALUATION_PARAMETER
CURR_METRIC = 'NDCG'

def dict_to_sorted_str(d: dict) -> str:
    return '@'.join(f"{k}={d[k]}" for k in sorted(d))


def get_best_validation_combos():
    combos = []
    for dataset in get_datasets(datasets=DATASETS):
        ds = dataset.get_name()
        for rec in get_recommenders(recommenders=RECOMMENDERS):
            rec_name = rec.get_name()
            csv = os.path.join(get_metrics_filepath(kw.VALIDATION, ds, rec_name), 'metrics.csv')
            if not os.path.exists(csv):
                continue
            df = pd.read_csv(csv, sep=kw.DELIMITER)
            if df.empty or 'Parameters' not in df or BEST_COLUMN not in df:
                continue
            rec_keys = set(rec.get_recommender_hyperparameters().keys())
            emb_keys = set(rec.get_embeddings_hyperparameters().keys())
            rows = []
            for _, row in df.iterrows():
                params = str_to_dict(row['Parameters'])
                rec_params = {k: v for k, v in params.items() if k in rec_keys}
                emb_params = {k: v for k, v in params.items() if k in emb_keys}
                rec_group = dict_to_sorted_str(rec_params) if rec_params else 'default'
                rows.append({
                    'rec_group': rec_group,
                    'rec_params': rec_params,
                    'embedding_params': emb_params,
                    'params_str': row['Parameters'],
                    BEST_COLUMN: row[BEST_COLUMN],
                    'embeddings_name': rec.get_embeddings_name(),
                    'recommender': rec_name,
                    'dataset': ds,
                    'model_cls': rec.get_model(),
                    'emb_model_cls': rec.get_embeddings_model(),
                })
            if not rows:
                continue
            ex = pd.DataFrame(rows)
            best = ex.loc[ex.groupby('rec_group')[BEST_COLUMN].idxmax()]
            combos.extend(best.to_dict('records'))
    return combos

def retrain_and_evaluate_test(combos):
    for c in combos:
        ds = c['dataset']
        df = next(get_datasets(datasets=[ds])).get_dataframe()
        df = remove_single_interactions(df)
        if kw.COLUMN_TIMESTAMP in df.columns:
            df[kw.COLUMN_DATETIME] = pd.to_datetime(df[kw.COLUMN_TIMESTAMP], unit='s')
        elif kw.COLUMN_DATETIME in df.columns:
            df[kw.COLUMN_DATETIME] = pd.to_datetime(df[kw.COLUMN_DATETIME]).dt.floor('s')
        else:
            raise ValueError('Coluna temporal não encontrada')
        df = df.sort_values(by=kw.COLUMN_DATETIME)

        from sklearn.model_selection import train_test_split
        df_train, df_rem = train_test_split(df, test_size=0.2, shuffle=False)
        df_val_aux, df_test = train_test_split(df_rem, test_size=0.5, shuffle=False)
        df_train = pd.concat([df_train, df_val_aux], axis=0)
        df_test = remove_cold_start(df_train, df_test)

        emb_path = get_embeddings_filepath(kw.TEST, ds, c['embeddings_name'], c['embedding_params'])
        rec_dir = get_recomendation_filepath(kw.TEST, ds, c['recommender'], c['rec_params'])
        metrics_dir = os.path.join(get_metrics_filepath(kw.TEST, ds, c['recommender']), c['rec_group'])

        # Ensure the 'epochs' parameter is correctly named before model instantiation
        if 'epochs' in c['embedding_params']:
            c['embedding_params']['epochs'] = c['embedding_params'].pop('epochs')

        emb_model = c['emb_model_cls'](emb_path, **c['embedding_params'])
        emb_model.fit(df_train)

        model = c['model_cls'](
            embeddings_filepath=emb_path,
            use_norm=c['rec_params'].get('recomender_norm', False),
            combination_strategy=c['rec_params'].get('combination_strategy', 'sum')
        )
        model.fit(df_train)
        recs = model.recommend(df_test)

        log_recommendations(rec_dir, c['params_str'], df_test, recs)
        m = Metrics(kw.N_EVAL)
        m.add_metrics(rec_dir)
        os.makedirs(metrics_dir, exist_ok=True)
        m.get_dataframe().to_csv(os.path.join(metrics_dir, 'metrics.csv'), sep=kw.DELIMITER, index=False)


def generate_grouped_graphs():
    for dataset in get_datasets(datasets=DATASETS):
        ds = dataset.get_name()
        rec_paths = {}
        groups = set()
        for rec in get_recommenders(recommenders=RECOMMENDERS):
            rec_name = rec.get_name()
            root = get_metrics_filepath(kw.TEST, ds, rec_name)
            if not os.path.exists(root):
                continue
            rec_paths[rec_name] = root
            for d in os.listdir(root):
                sub = os.path.join(root, d)
                if os.path.isdir(sub) and os.path.exists(os.path.join(sub, 'metrics.csv')):
                    groups.add(d)
        groups = sorted(groups)
        if not groups:
            continue

        rec_colors = {
            'Item2Vec_itemSim': 'blue',
            'TimeI2V_Disc_Aug': 'red',
            'TimeI2V_Cont': 'green',
        }

        chunk_size = 6
        for gstart in range(0, len(groups), chunk_size):
            chunk = groups[gstart:gstart + chunk_size]
            titles = []
            for group in chunk:
                try:
                    rec_params_str = dict_to_sorted_str(str_to_dict(group)).replace('@', ' · ')
                except Exception:
                    rec_params_str = group
                titles.append(rec_params_str)

            fig = make_subplots(rows=3, cols=2, subplot_titles=titles,
                                horizontal_spacing=0.08, vertical_spacing=0.12)

            legend_added = set()
            for idx, group in enumerate(chunk):
                row = idx // 2 + 1
                col = idx % 2 + 1
                for rec in get_recommenders(recommenders=RECOMMENDERS):
                    rec_name = rec.get_name()
                    root = rec_paths.get(rec_name)
                    if not root:
                        continue
                    csv = os.path.join(root, group, 'metrics.csv')
                    if not os.path.exists(csv):
                        continue
                    df = pd.read_csv(csv, sep=kw.DELIMITER)
                    if df.empty or 'Parameters' not in df.columns:
                        continue
                    rec_keys = set(rec.get_recommender_hyperparameters().keys())
                    df['Embedding'] = df['Parameters'].apply(
                        lambda p: dict_to_sorted_str({k: v for k, v in str_to_dict(p).items() if k not in rec_keys}) or 'emb_default'
                    )
                    ndcg_cols = [c for c in df.columns if c.startswith(f'{CURR_METRIC}@')]
                    if not ndcg_cols:
                        continue
                    x_vals = [int(c.split('@')[1]) for c in ndcg_cols]
                    plot_df = df[['Embedding'] + ndcg_cols].drop_duplicates().set_index('Embedding')
                    if plot_df.empty:
                        continue

                    color = rec_colors.get(rec_name, None)
                    show_leg = rec_name not in legend_added
                    for emb_name in plot_df.index:
                        fig.add_trace(
                            go.Scatter(
                                x=x_vals,
                                y=plot_df.loc[emb_name].values,
                                mode='lines+markers',
                                name=rec_name,
                                line=dict(color=color),
                                marker=dict(color=color),
                                legendgroup=rec_name,
                                showlegend=show_leg
                            ),
                            row=row, col=col
                        )
                    if show_leg:
                        legend_added.add(rec_name)

            fig.update_layout(
                title=f"{ds} | {CURR_METRIC}",
                height=900,
                width=1600,
                margin=dict(l=60, r=60, t=80, b=60),
                legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
                font=dict(family='Helvetica', size=12, color='black')
            )
            os.makedirs('figures', exist_ok=True)
            out_file = os.path.join('figures', f'{ds}_ALL_{CURR_METRIC}_subplots_{(gstart // chunk_size) + 1}.html')
            fig.write_html(out_file)
            webbrowser.open_new_tab('file://' + os.path.abspath(out_file))


if __name__ == '__main__':
    combos = get_best_validation_combos()
    retrain_and_evaluate_test(combos)
    generate_grouped_graphs()
