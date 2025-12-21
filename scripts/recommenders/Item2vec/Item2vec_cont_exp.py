import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# PyTorch imports
import torch
import scripts as kw

from .Data_repr import DataRepr
from .Item2Vec_abc import Item2vec_abstract
from .torchmodules.pytorch_dataset import create_item2vec_dataloader
from .torchmodules.pytorch_trainer import Item2VecTrainer
from .torchmodules.pytorch_model import Item2VecModel

class Item2vec_Exp(Item2vec_abstract):
    def __init__(self, embedding_dir, factors=100, w_size=-1, learning_rate=0.25, 
                 min_learning_rate=0.0025, subsample=0.0001, batch_size=kw.MEM_SIZE_LIMIT, 
                 negative_samples=5, negative_exp=0.75, min_weight=0.3, 
                 decay_rate=2, # Now acts as 'decay_rate' for Plateau Exp mode
                 epochs=100, min_time_diff=60, weight_floor=0.3, lr_decay=0.96, 
                 regularization=-1, recomender_norm=True):
        
        super().__init__(embedding_dir, factors, w_size, learning_rate, min_learning_rate, 
                         subsample, batch_size, negative_samples, negative_exp, epochs, 
                         lr_decay, regularization)
        
        self.min_time_diff = min_time_diff
        self.min_weight = min_weight
        self.decay_rate = decay_rate
        self.weight_floor = weight_floor
    
    def timestamp_cum(self, df):
        """
        Fast timestamp cumulative calculation using vectorized operations with scaler.
        """
        if kw.COLUMN_TIMESTAMP in df.columns:
            df = df.copy()
            df[kw.COLUMN_DATETIME] = pd.to_datetime(df[kw.COLUMN_TIMESTAMP], unit='s')
        elif kw.COLUMN_DATETIME in df.columns:
            df = df.copy()
            df[kw.COLUMN_DATETIME] = pd.to_datetime(df[kw.COLUMN_DATETIME])
        
        # Sort once
        df = df.sort_values([kw.COLUMN_USER_ID, kw.COLUMN_DATETIME])
        
        # Calculate time differences
        df[kw.COLUMN_TIME_DIFF] = df.groupby(kw.COLUMN_USER_ID)[kw.COLUMN_DATETIME].diff().dt.total_seconds().fillna(0).astype('int32')
        
        # Create mask for valid differences (non-noise)
        valid_mask = df[kw.COLUMN_TIME_DIFF] > self.min_time_diff
        
        # Create a temporary column with only valid differences
        df['valid_diffs'] = df[kw.COLUMN_TIME_DIFF].where(valid_mask)
        
        # Calculate Q1, Q3 and IQR using transform (vectorized)
        q1 = df.groupby(kw.COLUMN_USER_ID)['valid_diffs'].transform('quantile', 0.25).fillna(0)
        q3 = df.groupby(kw.COLUMN_USER_ID)['valid_diffs'].transform('quantile', 0.75).fillna(0)
        iqr = q3 - q1
        
        # Calculate mean and std from valid diffs only (vectorized)
        df[kw.COLUMN_MEAN] = df.groupby(kw.COLUMN_USER_ID)['valid_diffs'].transform('mean').fillna(0)
        df[kw.COLUMN_STD] = df.groupby(kw.COLUMN_USER_ID)['valid_diffs'].transform('std').fillna(0)
        
        # Clip time_diff using Q3 + 1.5 * IQR (vectorized)
        threshold = 1.5
        upper_clip = q3 + (threshold * iqr)
        df[kw.COLUMN_TIME_DIFF] = df[kw.COLUMN_TIME_DIFF].clip(upper=upper_clip)
        
        # Calculate cumulative sum per user (vectorized)
        df[kw.COLUMN_TIME_CUMSUM] = df.groupby(kw.COLUMN_USER_ID)[kw.COLUMN_TIME_DIFF].cumsum()
        
        # Apply MinMaxScaler normalization per user group using transform
        def scale_group(group):
            if len(group) == 1:
                return pd.Series([self.min_weight], index=group.index)
            
            scaler = MinMaxScaler(feature_range=(self.min_weight, 1))
            scaled = scaler.fit_transform(group.values.reshape(-1, 1)).flatten()
            return pd.Series(scaled, index=group.index)
        
        df[kw.COLUMN_TIME_CUMSUM_NORM] = (
            df.groupby(kw.COLUMN_USER_ID)[kw.COLUMN_TIME_CUMSUM]
            .transform(scale_group)
        )
        
        df = df.drop(columns=['valid_diffs'])
        
        return df

    def _calculate_plateau_exp_weights(self, distances, mean, std):
        """
        Plateau Exponential Decay.
        - Distance <= Mean: Weight = 1.0 (Plateau)
        - Distance > Mean:  Weight decays exponentially based on Std Dev.
        """
        # 1. Calculate Excess Distance (Shift)
        # Anything closer than the mean becomes 0 excess
        excess_dist = np.maximum(0, distances - mean)
        
        # 2. Normalize by Standard Deviation (Adaptive Scale)
        safe_std = np.where(std == 0, 1e-9, std)
        normalized_excess = excess_dist / safe_std
        
        # 3. Exponential Decay
        # curve_exp acts as the lambda decay rate here
        weights = np.exp(-self.decay_rate * normalized_excess)
        
        return np.round(weights, 2)

    def _calculate_linear_weights(self, norm_weights_i, norm_weights_context):
        """Calculates weights using Logarithmic Linear decay on normalized time."""
        norm_diff = np.abs(norm_weights_i - norm_weights_context)
        
        aux = 1 - norm_diff
        aux = np.maximum(aux, 1e-9) 
        
        log_term = np.log10(1 / aux)
        
        weights = 1 - log_term
        weights = np.maximum(weights, self.weight_floor)
        
        return np.round(weights, 2)

    def _generate_positive_data(self):
        """Generate only window-based positive samples (no full-pair mode)."""
        X_target, X_context, sample_weights = [], [], []

        for user_id in range(len(self.interaction_list)):
            curr_user = np.array(self.interaction_list[user_id])
            cumulative_time = np.array(self.cumsum_list[user_id])
            norm_weights = np.array(self.norm_weight_list[user_id])
            mean = self.mean_list[user_id]
            std = self.std_list[user_id]
            user_size = len(curr_user)

            if user_size < 2:
                continue

            for i in range(user_size):
                start_idx = max(0, i - self.window_size)
                end_idx = min(user_size, i + self.window_size + 1)
                context_indices = np.arange(start_idx, end_idx)

                X_target.extend(np.repeat(curr_user[i], len(context_indices)))
                X_context.extend(curr_user[context_indices])

                # Selection Logic based on curve_exp
                if self.decay_rate == -1:
                    # Linear (Log-Linear) Mode
                    linear_w = self._calculate_linear_weights(norm_weights[i], norm_weights[context_indices])
                    sample_weights.extend(linear_w)
                else:
                    # NEW: Plateau Exponential Mode
                    dist = np.abs(cumulative_time[i] - cumulative_time[context_indices])
                    exp_w = self._calculate_plateau_exp_weights(dist, mean, std)
                    sample_weights.extend(exp_w)

        print("\nNumber of samples:", len(X_target))
        print("Number of negative samples:", len(X_target) * self.negative_samples)
        self.steps_per_epoch = (len(X_target) // self.batch_size) + 1

        return np.array(X_target), np.array(X_context), np.array(sample_weights)

    def _fit_data(self, df):

        df = self._subsample_items(df)
        self.data_repr = DataRepr(df, temporal_sorting=True)

        self.interaction_list = self.data_repr.create_column_list(df, kw.COLUMN_ITEM_ID, transform=True)
        self.cumsum_list = self.data_repr.create_column_list(df, kw.COLUMN_TIME_CUMSUM)
        self.norm_weight_list = self.data_repr.create_column_list(df, kw.COLUMN_TIME_CUMSUM_NORM)
        self.mean_list = self.data_repr.create_metrics_list(df, kw.COLUMN_MEAN)
        self.std_list = self.data_repr.create_metrics_list(df, kw.COLUMN_STD)

        vocab_size = self.data_repr.get_n_items()
        
        self.item_freq = list(df.groupby(kw.COLUMN_ITEM_ID).size().values)
        self.cumulative_table = self._cumulative_table(self.item_freq)
        
        X_target, X_context, sample_weights = self._generate_positive_data()

        self.model = Item2VecModel(
            vocab_size=vocab_size,
            embedding_size=self.embedding_size, 
            learning_rate=self.learning_rate, 
            lr_decay=self.lr_decay, 
            regularization=self.regularization,
            loss_sum=True,
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from scripts.dataset import get_datasets 
    
    # Configuration
    TARGET_DATASET_NAME = 'kuaisim' 
    TARGET_USER_IDX = 10 
    
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
    valid_users = user_counts[user_counts > 20].index
    
    if len(valid_users) == 0:
        raise ValueError("No users with > 20 interactions found in dataset.")
        
    selected_user_id = valid_users[0]
    df_user = df_real[df_real[kw.COLUMN_USER_ID] == selected_user_id].copy()

    # --- Initialize Models ---
    # Model 1: Log-Linear (-1)
    model_lin = Item2vec_Exp(
        embedding_dir="tmp", 
        decay_rate=-1, 
        min_time_diff=60, 
        weight_floor=0.3
    )
    
    # Model 2: Plateau Exponential (New Logic)
    # Using curve_exp=1.0 as decay rate
    model_plat = Item2vec_Exp(
        embedding_dir="tmp", 
        decay_rate=0.1, 
        min_time_diff=60, 
        weight_floor=0.3
    )

    print("Processing timestamps...")
    df_processed = model_lin.timestamp_cum(df_user)
    
    cumulative_times = df_processed[kw.COLUMN_TIME_CUMSUM].values
    norm_times = df_processed[kw.COLUMN_TIME_CUMSUM_NORM].values
    mean_val = df_processed[kw.COLUMN_MEAN].iloc[0]
    std_val = df_processed[kw.COLUMN_STD].iloc[0]
    
    print(f"User Mean Gap: {mean_val:.2f}, Std: {std_val:.2f}")

    # Anchor Item (Middle)
    idx_mid = len(cumulative_times) // 2
    anchor_time = cumulative_times[idx_mid]
    anchor_norm = norm_times[idx_mid]

    # --- Calculate Weights Manually for Plotting ---
    
    # 1. Plateau Exp Weights
    dist = np.abs(cumulative_times - anchor_time)
    excess = np.maximum(0, dist - mean_val)
    safe_std = np.where(std_val == 0, 1e-9, std_val)
    norm_excess = excess / safe_std
    w_plat = np.exp(-1.0 * norm_excess)
    w_plat = np.round(w_plat, 2)
    
    # 2. Linear Weights
    norm_diff = np.abs(norm_times - anchor_norm)
    aux = 1 - norm_diff
    aux = np.maximum(aux, 1e-9)
    w_lin = 1 - np.log10(1/aux)
    w_lin = np.maximum(w_lin, 0.3)
    w_lin = np.round(w_lin, 2)

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(cumulative_times, w_lin, marker='.', label='Log-Linear (-1)', color='blue', alpha=0.5)
    plt.plot(cumulative_times, w_plat, marker='.', label='Plateau Exponential (exp=1)', color='red', alpha=0.5)
    
    plt.axvline(x=anchor_time, color='black', linestyle='--', label='Anchor Item')
    
    # Visualize the "Plateau" width (Mean range)
    # Highlight the area where dist <= mean
    plateau_start = anchor_time - mean_val
    plateau_end = anchor_time + mean_val
    plt.axvspan(plateau_start, plateau_end, color='red', alpha=0.1, label='Plateau Range (Weight=1.0)')

    plt.title(f'New Improvement: Linear vs Plateau Exponential\nUser {selected_user_id} (Mean Gap: {mean_val:.0f}s)', fontsize=14)
    plt.xlabel('Time (s)')
    plt.ylabel('Weight')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()