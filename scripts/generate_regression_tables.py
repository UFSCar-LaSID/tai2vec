import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

pd.options.display.max_colwidth = 500

best_column = 'rmse'

main_path = "results/metrics/test"
main_file = os.listdir(main_path)

dataframes = []

skip_recommenders = {'ALS', 'ALS_itemSim', 'BPR'}

for dataset_name in main_file:
    recommender_files = os.listdir(os.path.join(main_path, dataset_name))
    for recommender_name in recommender_files:
        if recommender_name in skip_recommenders:
            continue

        metrics_path = os.path.join(main_path, dataset_name, recommender_name, "regression_metrics.csv")
        if not os.path.exists(metrics_path):
            continue

        metrics_aux = pd.read_csv(metrics_path)
        metrics_aux.columns = ['Dataset', 'Recommender', 'rmse', 'mae']
        dataframes.append(metrics_aux)

metrics_df = pd.concat(dataframes, ignore_index=True)

if best_column == 'Mean':
    metrics_df.insert(3, 'Mean', metrics_df.mean(axis=1))

if best_column in metrics_df.columns:
    saida = metrics_df.loc[
        metrics_df.reset_index().groupby(['Dataset', 'Recommender'])[best_column].idxmax()
    ].reset_index(drop=True)
else:
    raise Exception("Coluna '" + best_column + "' não encontrada")


# ── Funções auxiliares ──────────────────────────────────────────────────────────

dataset_names = {
    'amazon-beauty': 'Amazon-Beauty',
    'amazon-books': 'Amazon-Books',
    'amazon-games': 'Amazon-Games',
    'ciaodvd': 'CiaoDVD',
    'ml-100k': 'ML-100k',
    'ml-1m': 'ML-1M',
    'bestbuy': 'BestBuy',
    'retailrocket-transactions': 'RetailRocket',
}

recommender_names = {
    'TimeI2V_Disc_Aug': 'TAI2Vec-Disc',
    'TimeI2V_Cont': 'TAI2Vec-Cont',
    'Item2Vec_itemSim': 'Item2Vec',
}

def pre_proc(df, comp_metric):
    df_aux = df.loc[:, ["Dataset", "Recommender", comp_metric]].copy()
    df_aux["Dataset"] = df_aux["Dataset"].replace(dataset_names)
    df_aux["Recommender"] = df_aux["Recommender"].replace(recommender_names)
    df_pivot = df_aux.pivot(index='Dataset', columns='Recommender', values=comp_metric)
    df_pivot = df_pivot.reset_index()
    return df_pivot


def create_gray_scale(df_pivot):
    GS = pd.concat([
        pd.DataFrame(
            MinMaxScaler(feature_range=(0.5, 0.95))
            .fit_transform(df_pivot.iloc[:, 1:].values.T).round(4).T
        ),
    ], axis=1)
    GS.index = df_pivot.index
    GS.columns = df_pivot.iloc[:, 1:].columns
    return GS


def add_gain(df_optimal, df_real):
    real_metrics = df_real.values[:, 1:]
    optimal_metrics = df_optimal.values[:, 1:]

    gain_percentage = (
        ((optimal_metrics - real_metrics) / real_metrics) * 100
    ).astype(float).round(1).astype(str)

    df_real.iloc[:, 2:4] = (
        df_real.round(4).astype(str).iloc[:, 2:4] + " (+" + gain_percentage[:, 1:3] + "\\%)"
    )
    df_real.iloc[:, 5:] = (
        df_real.round(4).astype(str).iloc[:, 5:] + " (+" + gain_percentage[:, 4:] + "\\%)"
    )
    return df_real


# ── Tabela NDCG@10 e NDCG@20 com escala de cinza ──────────────────────────────

# Escolha quais recommenders aparecem na tabela
selected_recommenders = ['Item2Vec_itemSim', 'Seq2Vec', 'TimeI2V_Cont', 'TimeI2V_Disc_Aug']

for comp_metric in ['mae', 'rmse']:
    df_final = pre_proc(
        saida[saida['Recommender'].isin(selected_recommenders)],
        comp_metric,
    )
    print(df_final)
    GS = create_gray_scale(df_final)

    str_cols = df_final.columns[1:]
    df_final[str_cols] = (
        "\\cellcolor[gray]{" + GS.astype(str) + "}"
        + df_final[str_cols].round(4).astype(str)
    )

    tabela_latex = (
        df_final.style.hide(axis="index")
        .to_latex(
            column_format='l' + 'c' * (df_final.shape[1] - 1),
            multicol_align='c',
            hrules='True',
            position_float='centering',
            caption=f"Comparação de recomendadores para {comp_metric}",
        )
    )
    tabela_latex = tabela_latex.replace(
        "\\begin{tabular}", "\\resizebox{\\textwidth} {!}{ \\begin{tabular}"
    )
    tabela_latex = tabela_latex.replace(
        "\\end{tabular}", "\\end{tabular} }"
    )
    print(tabela_latex)


# ── Tabela de comparação percentual contra Item2Vec (múltiplas métricas) ──────

comp_metrics = ['mae', 'rmse']
baseline_col = 'Item2Vec_itemSim'
exclude_recommenders = ['ALS', 'ALS_itemSim', 'BPR', baseline_col]

# Recommender display order and names for the comparison table
comp_recommender_order = ['Seq2Vec', 'TimeI2V_Cont', 'TimeI2V_Disc_Aug']
comp_recommender_names = {
    'Seq2Vec': 'Seq2Vec',
    'TimeI2V_Cont': 'TAI-Cont',
    'TimeI2V_Disc_Aug': 'TAI-Disc',
}

# Dataset display order
comp_dataset_order = [
    'amazon-beauty', 
    'amazon-books', 
    #'amazon-games', 
    'bestbuy', 
    'ciaodvd',
    'ml-100k', 
    #'ml-1m', 
    'retailrocket-transactions',
]

all_gain_dfs = {}
for metric in comp_metrics:
    df_aux = saida.loc[:, ["Dataset", "Recommender", metric]].copy()
    df_aux = df_aux[~df_aux['Recommender'].isin(exclude_recommenders)]

    df_pivot = df_aux.pivot(index='Dataset', columns='Recommender', values=metric)
    df_pivot = df_pivot.rename_axis(None, axis=1).reset_index()

    # Merge baseline back
    baseline = saida.loc[saida['Recommender'] == baseline_col, ["Dataset", metric]].copy()
    baseline = baseline.rename(columns={metric: 'baseline'})
    df_pivot = df_pivot.merge(baseline, on='Dataset')

    # Compute percentage gain
    for col in comp_recommender_order:
        if col in df_pivot.columns:
            df_pivot[col] = ((df_pivot[col] - df_pivot['baseline']) / df_pivot['baseline'] * 100).round(2)

    df_pivot = df_pivot.drop(columns=['baseline'])
    all_gain_dfs[metric] = df_pivot

# Build the combined table
rows = []
for ds in comp_dataset_order:
    row = {'Dataset': dataset_names.get(ds, ds)}
    for metric in comp_metrics:
        df_m = all_gain_dfs[metric]
        ds_row = df_m[df_m['Dataset'] == ds]
        for rec in comp_recommender_order:
            col_name = f"{comp_recommender_names[rec]}_{metric}"
            if not ds_row.empty and rec in ds_row.columns:
                row[col_name] = f"{ds_row[rec].values[0]:+.2f}\\%"
            else:
                row[col_name] = "N/A"
    rows.append(row)

# Compute mean and median
mean_row = {'Dataset': '\\textbf{Mean}'}
median_row = {'Dataset': '\\textbf{Median}'}
for metric in comp_metrics:
    df_m = all_gain_dfs[metric]
    # Filter to ordered datasets only
    df_m_filtered = df_m[df_m['Dataset'].isin(comp_dataset_order)]
    for rec in comp_recommender_order:
        col_name = f"{comp_recommender_names[rec]}_{metric}"
        if rec in df_m_filtered.columns:
            mean_row[col_name] = f"{df_m_filtered[rec].mean():.2f}\\%"
            median_row[col_name] = f"{df_m_filtered[rec].median():.2f}\\%"
        else:
            mean_row[col_name] = "N/A"
            median_row[col_name] = "N/A"

# Build LaTeX manually for exact formatting
metric_labels = {
    'NDCG@10': 'NDCG@10',
    'Hit_Rate@10': 'Hit Rate@10',
    'rmse': 'RMSE',
    'mae': 'MAE'
}
n_recs = len(comp_recommender_order)
n_metrics = len(comp_metrics)
total_cols = n_recs * n_metrics

# Header columns
rec_header = " & ".join([comp_recommender_names[r] for r in comp_recommender_order])
col_spec = "l " + " | ".join(["c" * n_recs] * n_metrics)

lines = []
lines.append("\\begin{table}[ht]")
lines.append("\\centering")
lines.append("\\caption{Percentage gain over Item2Vec for " + ", ".join(metric_labels[m] for m in comp_metrics) + "}")
lines.append("\\label{table:percentage_gain_combined}")
lines.append("\\resizebox{\\textwidth}{!}{")
lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
lines.append("\\toprule")

# Multicolumn metric headers
multi_parts = []
for metric in comp_metrics:
    multi_parts.append(f"\\multicolumn{{{n_recs}}}{{c}}{{\\textbf{{{metric_labels[metric]}}}}}")
lines.append("& " + " & ".join(multi_parts) + " \\\\")

# cmidrule
cmidrules = []
start = 2
for i in range(n_metrics):
    end = start + n_recs - 1
    cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
    start = end + 1
lines.append(" ".join(cmidrules))

# Sub-header with recommender names
sub_header = " & ".join([rec_header] * n_metrics)
lines.append(f"Dataset & {sub_header} \\\\")
lines.append("\\midrule")

# Data rows
for row in rows:
    vals = []
    for metric in comp_metrics:
        for rec in comp_recommender_order:
            col_name = f"{comp_recommender_names[rec]}_{metric}"
            vals.append(row[col_name])
    lines.append(f"{row['Dataset']} & " + " & ".join(vals) + " \\\\")

# Midrule + mean/median
lines.append("\\midrule")
mean_vals = []
median_vals = []
for metric in comp_metrics:
    for rec in comp_recommender_order:
        col_name = f"{comp_recommender_names[rec]}_{metric}"
        mean_vals.append(mean_row[col_name])
        median_vals.append(median_row[col_name])
lines.append(f"\\textbf{{Mean}} & " + " & ".join(mean_vals) + " \\\\")
lines.append(f"\\textbf{{Median}} & " + " & ".join(median_vals) + " \\\\")

lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("}")
lines.append("\\end{table}")

tabela_comparacao = "\n".join(lines)
print(tabela_comparacao)