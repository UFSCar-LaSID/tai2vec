import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

from sklearn.preprocessing import MinMaxScaler
import torch
import scripts as kw

from .Data_repr import DataRepr
from .Item2Vec_abc import Item2vec_abstract
from .torchmodules.pytorch_dataset import create_item2vec_dataloader
from .torchmodules.pytorch_trainer import Item2VecTrainer
from .torchmodules.pytorch_model import Item2VecModel

class Item2vec_Temp_Cont_model(Item2vec_abstract):
    def __init__(self, embedding_dir, factors=100, w_size=-1, learning_rate=0.25, 
                 min_learning_rate=0.0025, subsample=0.0001, batch_size=kw.MEM_SIZE_LIMIT, 
                 negative_samples=5, negative_exp=0.75, min_weight=0.3, 
                 decay_rate=2, 
                 epochs=100, min_time_diff=60, weight_floor=0, lr_decay=0.96, 
                 regularization=-1, recomender_norm=True, big_innit=False):
        
        super().__init__(embedding_dir, factors, w_size, learning_rate, min_learning_rate, 
                         subsample, batch_size, negative_samples, negative_exp, epochs, 
                         lr_decay, regularization, recomender_norm, big_innit)
        
        self.min_time_diff = min_time_diff
        self.min_weight = min_weight
        self.decay_rate = decay_rate
        self.weight_floor = weight_floor
    
    def timestamp_cum(self, df):
        
        if kw.COLUMN_TIMESTAMP in df.columns:
            df = df.copy()
            df[kw.COLUMN_DATETIME] = pd.to_datetime(df[kw.COLUMN_TIMESTAMP], unit='s')
        elif kw.COLUMN_DATETIME in df.columns:
            df = df.copy()
            df[kw.COLUMN_DATETIME] = pd.to_datetime(df[kw.COLUMN_DATETIME])
        
        # 1. Sort & Diff
        df = df.sort_values([kw.COLUMN_USER_ID, kw.COLUMN_DATETIME])
        df[kw.COLUMN_TIME_DIFF] = df.groupby(kw.COLUMN_USER_ID)[kw.COLUMN_DATETIME].diff().dt.total_seconds().fillna(0).astype('int32')
        
        # 2. Filter Valid Gaps for Stats
        valid_mask = df[kw.COLUMN_TIME_DIFF] > self.min_time_diff
        df['valid_diffs'] = df[kw.COLUMN_TIME_DIFF].where(valid_mask)
        
        # 3. Calculate Limits (FIXED: fillna with Infinity)
        q1 = df.groupby(kw.COLUMN_USER_ID)['valid_diffs'].transform('quantile', 0.25)
        q3 = df.groupby(kw.COLUMN_USER_ID)['valid_diffs'].transform('quantile', 0.75)
        iqr = q3 - q1
        
        upper_clip = q3 + (1.5 * iqr)
        upper_clip = upper_clip.fillna(np.inf)
        
        # 4. Apply Clipping
        df[kw.COLUMN_TIME_DIFF] = df[kw.COLUMN_TIME_DIFF].clip(upper=upper_clip).infer_objects(copy=False)
        
        # 5. Stats & Cumsum
        df[kw.COLUMN_MEAN] = df.groupby(kw.COLUMN_USER_ID)['valid_diffs'].transform('mean').fillna(0)
        df[kw.COLUMN_STD] = df.groupby(kw.COLUMN_USER_ID)['valid_diffs'].transform('std').fillna(0)
        df[kw.COLUMN_TIME_CUMSUM] = df.groupby(kw.COLUMN_USER_ID)[kw.COLUMN_TIME_DIFF].cumsum()
        
        # 6. Normalize Time (0-1)
        def scale_group(group):
            if len(group) == 1:
                return pd.Series([self.min_weight], index=group.index)
            scaler = MinMaxScaler(feature_range=(self.weight_floor, 1))
            scaled = scaler.fit_transform(group.values.reshape(-1, 1)).flatten()
            return pd.Series(scaled, index=group.index)
        
        df[kw.COLUMN_TIME_CUMSUM_NORM] = (
            df.groupby(kw.COLUMN_USER_ID)[kw.COLUMN_TIME_CUMSUM]
            .transform(scale_group)
        )
        
        return df.drop(columns=['valid_diffs'])

    def _calculate_z_weights(self, distances, mean, std):

        safe_std = np.where(std == 0, 1e-9, std)
        z_score = (distances - mean) / safe_std
        z_score = np.where(std == 0, 0, z_score)
        z_score = np.clip(z_score, 0, 20)
        
        # Decay: 1 - (z/(z+1))^exp
        weights = np.where(z_score == 0, 1, 1 - np.power((z_score / (z_score + 1)), self.decay_rate))
        weights = np.maximum(weights, self.weight_floor)

        return np.round(weights, 2)

    def _calculate_linear_weights(self, norm_weights_i, norm_weights_context):

        norm_diff = np.abs(norm_weights_i - norm_weights_context)
        similarity = (1 - norm_diff)
        weights = np.maximum(similarity, self.weight_floor)
        
        return np.round(weights, 2)

    def _generate_positive_data(self):

        X_target, X_context, sample_weights = [], [], []

        for user_id in range(len(self.interaction_list)):
            curr_user = np.array(self.interaction_list[user_id])
            cumulative_time = np.array(self.cumsum_list[user_id])
            norm_weights = np.array(self.norm_weight_list[user_id])
            mean = self.mean_list[user_id]
            std = self.std_list[user_id]
            user_size = len(curr_user)

            if user_size < 2: continue

            for i in range(user_size):
                start_idx = max(0, i - self.window_size)
                end_idx = min(user_size, i + self.window_size + 1)
                context_indices = np.arange(start_idx, end_idx)
                
                context_indices = context_indices[context_indices != i]
                if len(context_indices) == 0: continue

                X_target.extend(np.repeat(curr_user[i], len(context_indices)))
                X_context.extend(curr_user[context_indices])

                if self.decay_rate == -1:
                    w1 = self._calculate_linear_weights(norm_weights[i], norm_weights[context_indices])
                    sample_weights.extend(w1)
                else:
                    dist = np.abs(cumulative_time[i] - cumulative_time[context_indices])
                    w1 = self._calculate_linear_weights(norm_weights[i], norm_weights[context_indices])
                    w2 = self._calculate_z_weights(dist, mean, std)
                    sample_weights.extend((w1+w2)/2)

        print("\nNumber of samples:", len(X_target))
        print("Number of negative samples:", len(X_target) * self.negative_samples)
        self.steps_per_epoch = (len(X_target) // self.batch_size) + 1

        return np.array(X_target), np.array(X_context), np.array(sample_weights)

    def _fit_data(self, df):

        if kw.COLUMN_TIMESTAMP in df.columns or kw.COLUMN_DATETIME in df.columns:
            df = self.timestamp_cum(df.copy())
        else:
            raise Exception("Timestamp column not found")

        self.data_repr = DataRepr(df, temporal_sorting=True)
        df = self._subsample_items(df)

        self.interaction_list = self.data_repr.create_column_list(df, kw.COLUMN_ITEM_ID, transform=True)
        self.cumsum_list = self.data_repr.create_column_list(df, kw.COLUMN_TIME_CUMSUM)
        self.norm_weight_list = self.data_repr.create_column_list(df, kw.COLUMN_TIME_CUMSUM_NORM)
        self.mean_list = self.data_repr.create_metrics_list(df, kw.COLUMN_MEAN)
        self.std_list = self.data_repr.create_metrics_list(df, kw.COLUMN_STD)
        
        X_target, X_context, sample_weights = self._generate_positive_data()

        self.model = Item2VecModel(
            vocab_size=self.vocab_size, 
            embedding_size=self.embedding_size, 
            learning_rate=self.learning_rate, 
            lr_decay=self.lr_decay, 
            regularization=self.regularization,
            loss_sum=True,
            big_innit=self.big_innit
        ).to('cuda' if torch.cuda.is_available() else 'cpu')

        max_workers = os.cpu_count()
        print(f"Using {max_workers} workers for DataLoader")
        
        dataloader = create_item2vec_dataloader(
            X_target=X_target, 
            X_context=X_context, 
            cumulative_table=self.cumulative_table, 
            negative_samples=self.negative_samples,
            batch_size=self.batch_size, 
            weights=sample_weights, 
            shuffle=True,
            num_workers=8
        )

        trainer = Item2VecTrainer(self, self.model)
        self.model = trainer.train(dataloader, self.data_repr)


def plot_weights_for_user(ax, df_user, user_id, model, plot_mode):
    """
    Plota os pesos temporais para um determinado usuário em um eixo de subplot.
    O eixo X está em HORAS.
    """
    df_processed = model.timestamp_cum(df_user)
    
    # ==========================
    # Tempo cumulativo
    # ==========================

    # Em segundos (para cálculo – não muda nada)
    cumulative_seconds = df_processed[kw.COLUMN_TIME_CUMSUM].values

    # 🔹 Para o eixo X: converter para HORAS
    cumulative_hours = cumulative_seconds / 84600.0

    norm_times = df_processed[kw.COLUMN_TIME_CUMSUM_NORM].values
    mean_val = df_processed[kw.COLUMN_MEAN].iloc[0]
    std_val = df_processed[kw.COLUMN_STD].iloc[0]

    indices_para_plotar = {
        'Primeiro Item': 0,
        'Item do Meio': len(cumulative_hours) // 2,
        'Último Item': len(cumulative_hours) - 1
    }

    cores = {
        'Primeiro Item': '#1f77b4',
        'Item do Meio': '#ff7f0e',
        'Último Item': '#2ca02c'
    }

    for titulo, idx in indices_para_plotar.items():
        anchor_time = cumulative_seconds[idx]

        # Distâncias continuam em segundos
        dist = np.abs(cumulative_seconds - anchor_time)

        w_z = model._calculate_z_weights(dist, mean_val, std_val)
        w_lin = model._calculate_linear_weights(norm_times[idx], norm_times)

        cor = cores[titulo]
        estilo_plot = {'marker': 'o', 'markersize': 6, 'linewidth': 2.5, 'alpha': 0.7}

        if plot_mode == 'log':
            ax.plot(cumulative_hours, w_lin, label=titulo, color=cor, **estilo_plot)
        elif plot_mode == 'z-score':
            ax.plot(cumulative_hours, w_z, label=titulo, color=cor, **estilo_plot)
        elif plot_mode == 'mean':
            w_mean = (w_lin + w_z) / 2
            ax.plot(cumulative_hours, w_mean, label=titulo, color=cor, **estilo_plot)

        # Linha vertical também em horas
        ax.axvline(
            x=cumulative_hours[idx],
            color=cor,
            linestyle='--',
            alpha=0.8,
            linewidth=2
        )

    title_map = {
        'z-score': 'Z-Score',
        'log': 'Log-Linear',
        'mean': 'Média (Z-Score + Log-Linear)'
    }

    ax.set_title(f'Parâmetro de decaimento = {model.decay_rate}', fontsize=20)
    ax.set_xlabel('Tempo acumulado (dias)', fontsize=16)
    ax.set_ylabel('Peso', fontsize=16)
    ax.legend(title='Item alvo')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.2, 1.05)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from scripts.dataset import get_datasets 
    
    PLOT_MODE = 'mean'
    
    TARGET_DATASET_NAME = 'amazon-books' 
    
    print(f"Loading real dataset: {TARGET_DATASET_NAME}...")
    target_dataset = None
    for ds in get_datasets(datasets=[TARGET_DATASET_NAME]):
        if ds.get_name() == TARGET_DATASET_NAME:
            target_dataset = ds
            break
            
    if target_dataset is None:
        raise ValueError(f"Dataset '{TARGET_DATASET_NAME}' not found/loaded.")

    df_real = target_dataset.get_dataframe()
    user_counts = df_real[kw.COLUMN_USER_ID].value_counts()

    try:
        user_50_items_id = user_counts[user_counts == 50].index[0]
    except IndexError:
        raise ValueError("Não foi encontrado usuário com 50 itens.")

    df_user_50 = df_real[df_real[kw.COLUMN_USER_ID] == user_50_items_id].copy()

    # --- INÍCIO DA ALTERAÇÃO ---
    # Cria dois modelos com decay_rate diferentes
    model_decay_3 = Item2vec_Temp_Cont_model(
        embedding_dir="tmp", 
        decay_rate=3, 
        min_time_diff=300, 
        weight_floor=0.3
    )
    
    model_decay_5 = Item2vec_Temp_Cont_model(
        embedding_dir="tmp", 
        decay_rate=5, 
        min_time_diff=300, 
        weight_floor=0.3
    )

    # Cria os subplots 1x2
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plota para o modelo com decay_rate=3 no subplot da esquerda
    plot_weights_for_user(
        axes[0],
        df_user_50,
        user_50_items_id,
        model_decay_3,
        PLOT_MODE
    )
    
    # Plota para o modelo com decay_rate=5 no subplot da direita
    plot_weights_for_user(
        axes[1],
        df_user_50,
        user_50_items_id,
        model_decay_5,
        PLOT_MODE
    )

    plt.tight_layout()
    plt.show()
    # --- FIM DA ALTERAÇÃO ---