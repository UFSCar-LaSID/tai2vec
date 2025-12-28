import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import numpy as np
import pandas as pd
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
                 epochs=100, min_time_diff=60, weight_floor=0.3, lr_decay=0.96, 
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
        df[kw.COLUMN_TIME_DIFF] = df[kw.COLUMN_TIME_DIFF].clip(upper=upper_clip)
        
        # 5. Stats & Cumsum
        df[kw.COLUMN_MEAN] = df.groupby(kw.COLUMN_USER_ID)['valid_diffs'].transform('mean').fillna(0)
        df[kw.COLUMN_STD] = df.groupby(kw.COLUMN_USER_ID)['valid_diffs'].transform('std').fillna(0)
        df[kw.COLUMN_TIME_CUMSUM] = df.groupby(kw.COLUMN_USER_ID)[kw.COLUMN_TIME_DIFF].cumsum()
        
        # 6. Normalize Time (0-1)
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
        similarity = 1 - norm_diff
        similarity = np.maximum(similarity, 1e-9) 
        
        # Using log2 makes decay aggressive; switch to log10 for gentler slope if needed
        log_term = np.log2(1 / similarity)
        weights = 1 - log_term
        weights = np.maximum(weights, self.weight_floor)
        
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

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from scripts.dataset import get_datasets 
    
    TARGET_DATASET_NAME = 'kuaisim' 
    
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
    
    selected_user_id = valid_users[0]
    df_user = df_real[df_real[kw.COLUMN_USER_ID] == selected_user_id].copy()

    # Model Init (Only need 1 instance to process time)
    model = Item2vec_Temp_Cont_model(
        embedding_dir="tmp", 
        decay_rate=2, 
        min_time_diff=60, 
        weight_floor=0.3
    )

    print("Processing timestamps...")
    df_processed = model.timestamp_cum(df_user)
    
    cumulative_times = df_processed[kw.COLUMN_TIME_CUMSUM].values
    norm_times = df_processed[kw.COLUMN_TIME_CUMSUM_NORM].values
    mean_val = df_processed[kw.COLUMN_MEAN].iloc[0]
    std_val = df_processed[kw.COLUMN_STD].iloc[0]
    
    print(f"User Mean Gap: {mean_val:.2f}, Std: {std_val:.2f}")

    idx_mid = len(cumulative_times) // 2
    anchor_time = cumulative_times[idx_mid]
    anchor_norm = norm_times[idx_mid]
    
    dist = np.abs(cumulative_times - anchor_time)
    z_score = (dist - mean_val) / (std_val + 1e-9)
    z_score = np.clip(z_score, 0, 20)
    w_z = np.where(z_score == 0, 1, 1 - np.power((z_score / (z_score + 1)), 2)) # decay_rate=2
    w_z = np.round(w_z, 2)
    
    norm_diff = np.abs(norm_times - anchor_norm)
    similarity = 1 - norm_diff
    similarity = np.maximum(similarity, 1e-9)
    w_lin = 1 - np.log2(1/similarity)
    w_lin = np.maximum(w_lin, 0.3) # Floor
    w_lin = np.round(w_lin, 2)

    plt.figure(figsize=(14, 6))
    plt.plot(cumulative_times, w_lin, marker='.', label='Log-Linear (-1)', color='blue', alpha=0.5)
    plt.plot(cumulative_times, w_z, marker='.', label='Z-Score (exp=2)', color='red', alpha=0.5)
    
    plt.axvline(x=anchor_time, color='black', linestyle='--', label='Anchor Item')
    plt.title(f'Method Comparison: Linear vs Z-Score\nUser {selected_user_id}', fontsize=14)
    plt.xlabel('Time (s)')
    plt.ylabel('Weight')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()