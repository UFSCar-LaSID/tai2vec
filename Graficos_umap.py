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
warnings.filterwarnings('once')

best_column = kw.EVALUATION_PARAMETER
metric_type = ["Prec", "Rec", "F1_Score", "Hit_Rate", "NDCG"]
top_k = [3, 5, 10, 20]

# User configuration
selected_datasets = ['amazon-beauty', 'ml-100k', 'ciaodvd', 'amazon-books']
n_rows = 2  # Number of rows in the subplot
n_cols = 2  # Number of columns in the subplot

main_path = "results/metrics/test/"
all_datasets = os.listdir(main_path)
curr_metric = "NDCG"

# Filter for selected datasets that exist
main_file = [ds for ds in selected_datasets if ds in all_datasets]
if len(main_file) != len(selected_datasets):
    print("Warning: Some selected datasets were not found and will be skipped.")

dataframes = []           

for dataset_name in main_file:
        
    recommender_files = os.listdir(os.path.join(main_path, dataset_name))
    for recommender_name in recommender_files:
                
        metrics_path = os.path.join(main_path, dataset_name, recommender_name, "metrics.csv")
        metrics_aux = pd.read_csv(metrics_path, sep=';')
        
        metrics_aux.insert(0,'Recommender','')
        metrics_aux['Recommender'] = recommender_name

        metrics_aux.insert(0,'Dataset','')
        metrics_aux['Dataset'] = dataset_name
        
        dataframes.append(metrics_aux)
                
metrics_df = pd.concat(dataframes, ignore_index=True)

#Remove a coluna Mean
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

all_recommenders = saida['Recommender'].unique()
colors = px.colors.qualitative.Plotly
color_map = {rec: colors[i % len(colors)] for i, rec in enumerate(all_recommenders)}

fig = make_subplots(
    rows=n_rows, 
    cols=n_cols, 
    subplot_titles=main_file,
    x_title="", 
    y_title="",
    horizontal_spacing=0.1,
    vertical_spacing=0.1
)

marker_symbols = ['circle', 'square', 'diamond', 'cross', 'triangle-up', 'triangle-down', 'star', 'hexagram']

for i, dataset_name in enumerate(main_file):
    
    row = (i // n_cols) + 1
    col = (i % n_cols) + 1

    saida_aux = saida[saida['Dataset'] == dataset_name] 
        
    my_regex = "Recommender|" +  curr_metric + ".*"
    df_aux = saida_aux.filter(regex=(my_regex))
    
    df_aux = df_aux.set_index('Recommender')
    df_aux = df_aux.rename(columns=function_1)
    
    for j, recomendador in enumerate(df_aux.index):
        fig.add_trace(
            go.Scatter(
                x=df_aux.columns,
                y=df_aux.loc[recomendador],
                mode='lines+markers',
                marker=dict(symbol=marker_symbols[j % len(marker_symbols)], size=7, line=dict(color='black', width=0.5)),
                name=recomendador,
                legendgroup=recomendador,
                showlegend= (i == 0), # Show legend only for the first subplot
                line=dict(color=color_map[recomendador])
            ),
            row=row,
            col=col
        )

fig.update_layout(
    title_text=f"Metrics Comparison - {curr_metric}",
    title_x=0.5,
    height=400 * n_rows,
    width=600 * n_cols,
    legend_title="Recommenders",
    font=dict(
        family="Helvetica",
        size=12,
        color="black"
    ),
    plot_bgcolor='white'
)

fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

figure_name = "subplot_" + curr_metric + ".html"
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