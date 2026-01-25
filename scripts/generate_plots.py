
import sys
import os

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(parent_path)

import numpy as np

import pandas as pd
pd.options.plotting.backend = "plotly"

import os
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.preprocessing import MinMaxScaler
import scripts as kw

import warnings
import webbrowser

from scripts.modules.dataset import DATASETS_TABLE
from scripts.modules.recommenders import RECOMMENDERS_TABLE
from scripts.modules.utils.parameters_handle import get_input

warnings.filterwarnings('once')

best_column = kw.EVALUATION_PARAMETER
metric_type = ["Prec", "Rec", "F1_Score", "Hit_Rate", "NDCG"]
top_k = [3, 5, 10, 20]

main_path = "results/metrics/test/"
main_file = os.listdir(main_path)
curr_metric = "NDCG"

dataframes = []

dataset_options, recommender_options = get_input('Choose datasets to preprocess', [
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

for dataset_name in dataset_names:
        
    recommender_files = os.listdir(os.path.join(main_path, dataset_name))
    for recommender_name in recommender_names:
                
        metrics_path = os.path.join(main_path, dataset_name, recommender_name, "metrics.csv")
        metrics_aux = pd.read_csv(metrics_path, sep=';')
        
        metrics_aux.insert(0,'Recommender','')
        metrics_aux['Recommender'] = recommender_name

        metrics_aux.insert(0,'Dataset','')
        metrics_aux['Dataset'] = dataset_name
        
        dataframes.append(metrics_aux)
                
metrics_df = pd.concat(dataframes, ignore_index=True)

metrics_df = metrics_df[metrics_df.Recommender != 'ALS_mean']
metrics_df = metrics_df[metrics_df.Recommender != 'BPR_mean']

if (best_column == 'Mean'):
    metrics_df.insert(3,'Mean', metrics_df.mean(axis=1))
    
if best_column in metrics_df.columns:
    saida = metrics_df.loc[metrics_df.reset_index().groupby(['Dataset', 'Recommender'])[best_column].idxmax()].reset_index(drop=True)
else:
    raise Exception("Coluna '" + best_column + "' não encontrada")

if not os.path.exists("figures"):
    os.mkdir("figures")

def function_1(x):
    if(len(x.split('@')) > 1): 
        x = x.split('@')[1]
    return x

for dataset_name in main_file:
    
    saida_aux = saida[saida['Dataset'] == dataset_name] 
        
    my_regex = "Recommender|" +  curr_metric + ".*"
    df_aux = saida_aux.filter(regex=(my_regex))
    
    df_aux = df_aux.set_index('Recommender')
    df_aux = df_aux.rename(columns=function_1)
    
    marker_symbols = ['circle', 'square', 'diamond', 'cross', 'triangle-up', 'triangle-down', 'star', 'hexagram']

    fig = go.Figure()

    for i, recomendador in enumerate(df_aux.index):
        fig.add_trace(
            go.Scatter(
                x=df_aux.columns,
                y=df_aux.loc[recomendador], 
                mode='lines+markers',
                marker=dict(symbol=marker_symbols[i], size=7, line=dict(color='black', width=0.5)),
                name=recomendador
            )
        )

    fig.update_layout(
    title=dataset_name+" - "+curr_metric,
    title_x=0.15,
    title_y=0.8,
    showlegend=True,
    height=400,
    width=600,
    yaxis_title="",
    xaxis_title="",
    legend_title="Recommenders",
    font=dict(
        family="Helvetica",
        size=12,
        color="black"
        )
    )

    figure_name = dataset_name + "_" + curr_metric + ".html"
    fig.write_html(os.path.join("figures", figure_name)) 
    html_path = os.path.abspath(os.path.join("figures", figure_name))
    try:
        browser = webbrowser.get('chrome')
    except webbrowser.Error:
        try:
            browser = webbrowser.get('google-chrome')
        except webbrowser.Error:
            browser = None
    if browser:
        browser.open(f'file://{html_path}')