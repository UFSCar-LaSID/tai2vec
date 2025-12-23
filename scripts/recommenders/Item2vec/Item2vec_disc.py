import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import numpy as np
import pandas as pd
import torch
import scripts as kw

from .Data_repr import DataRepr
from .Item2Vec_abc import Item2vec_abstract
from .torchmodules.pytorch_dataset import create_item2vec_dataloader
from .torchmodules.pytorch_trainer import Item2VecTrainer
from .torchmodules.pytorch_model import Item2VecModel

class Item2vec_temp_model(Item2vec_abstract):

    def __init__(self, embedding_dir, factors=50, w_size=-1, learning_rate=0.25, 
                 min_learning_rate=0.0025, subsample=0.001, batch_size=kw.MEM_SIZE_LIMIT, 
                 negative_samples=3, negative_exp=0.75, epochs=100, lr_decay=0.1, 
                 time_exp=1.5, min_time_diff=300, regularization=-1, recomender_norm=True, big_innit=False):
        
        super().__init__(embedding_dir, factors, w_size, learning_rate, min_learning_rate, 
                         subsample, batch_size, negative_samples, negative_exp, epochs, 
                         lr_decay, regularization, recomender_norm, big_innit)
        
        self.time_exp = time_exp
        self.min_time_diff = min_time_diff

    def timestamp_diff(self, df):
        
        if kw.COLUMN_TIMESTAMP in df.columns:
            df = df.copy()
            df[kw.COLUMN_DATETIME] = pd.to_datetime(df[kw.COLUMN_TIMESTAMP], unit='s')
        elif kw.COLUMN_DATETIME in df.columns:
            df = df.copy()
            df[kw.COLUMN_DATETIME] = pd.to_datetime(df[kw.COLUMN_DATETIME])

        # 1. Sort & Diff
        df = df.sort_values([kw.COLUMN_USER_ID, kw.COLUMN_DATETIME])
        df[kw.COLUMN_TIME_DIFF] = df.groupby(kw.COLUMN_USER_ID)[kw.COLUMN_DATETIME].diff().dt.total_seconds().fillna(0).astype('int32')
    
        # 2. Filter valid gaps for stats calculation (ignore noise)
        valid_gaps_mask = df[kw.COLUMN_TIME_DIFF] > self.min_time_diff
        df['temp_diffs'] = df[kw.COLUMN_TIME_DIFF].where(valid_gaps_mask)

        # 3. Calculate Thresholds efficiently
        q1 = df.groupby(kw.COLUMN_USER_ID)['temp_diffs'].transform('quantile', 0.25)
        q3 = df.groupby(kw.COLUMN_USER_ID)['temp_diffs'].transform('quantile', 0.75)
        iqr = q3 - q1
        
        threshold = q3 + (self.time_exp * iqr)
        
        threshold = threshold.fillna(np.inf)
        
        # 4. Create Session Splits
        df['mask'] = df[kw.COLUMN_TIME_DIFF] >= threshold
        df['increment'] = df.groupby(kw.COLUMN_USER_ID)['mask'].cumsum()
        
        # 5. Create Pseudo-User IDs
        df['old_user_id'] = df[kw.COLUMN_USER_ID]
        df[kw.COLUMN_USER_ID] = df.groupby([kw.COLUMN_USER_ID, 'increment']).ngroup()
        
        # Cleanup
        df.drop(columns=['mask', 'temp_diffs'], inplace=True)
        
        return df

    def _generate_positive_data(self):
        X_target = []
        X_context = []

        # Iterate over pseudo-users (sessions)
        for user_id in range(len(self.interaction_list)):
            curr_user = np.array(self.interaction_list[user_id], dtype=np.int32)
            user_size = curr_user.shape[0]

            if user_size < 2:
                continue

            for i in range(user_size):
                start_idx = max(0, i - self.window_size)
                end_idx = min(user_size, i + self.window_size + 1)

                context_indices = np.arange(start_idx, end_idx)
                context_indices = context_indices[context_indices != i]

                if len(context_indices) == 0:
                    continue

                X_target.extend([curr_user[i]] * len(context_indices))
                X_context.extend(curr_user[context_indices])

        X_target = np.asarray(X_target, dtype=np.int32)
        X_context = np.asarray(X_context, dtype=np.int32)

        print(f"\nNumber of positive samples: {len(X_target):,}")
        print(f"Number of negative samples: {len(X_target) * self.negative_samples:,}")

        self.steps_per_epoch = (len(X_target) // self.batch_size) + (
            1 if len(X_target) % self.batch_size else 0)

        return X_target, X_context

    def _fit_data(self, df):

        self.item_freq = list(df.groupby(kw.COLUMN_ITEM_ID).size().values)
        self.cumulative_table = self._cumulative_table(self.item_freq)
        self.vocab_size = len(self.item_freq)

        # 1. Process Time & Split Sessions (Pseudo-Users created here)
        if kw.COLUMN_TIMESTAMP in df.columns or kw.COLUMN_DATETIME in df.columns:
            df = self.timestamp_diff(df.copy())
        else:
            raise Exception("Timestamp column not found")

        # 2. Subsample and Prepare Data
        self.data_repr = DataRepr(df)
        df = self._subsample_items(df)
        self.interaction_list = self.data_repr.create_interaction_list(df)
        
        # 3. Model Init
        self.model = Item2VecModel(
            vocab_size=self.vocab_size, 
            embedding_size=self.embedding_size, 
            learning_rate=self.learning_rate, 
            lr_decay=self.lr_decay, 
            regularization=self.regularization,
            loss_sum=True,
            big_innit=self.big_innit,
        ).to('cuda' if torch.cuda.is_available() else 'cpu')
        
        X_target, X_context = self._generate_positive_data()

        max_workers = os.cpu_count()
        print(f"Using {max_workers} workers for DataLoader")
        
        dataloader = create_item2vec_dataloader(
            X_target=X_target, 
            X_context=X_context, 
            cumulative_table=self.cumulative_table, 
            negative_samples=self.negative_samples,
            batch_size=self.batch_size, 
            weights=None, 
            shuffle=True,
            num_workers=8
        )

        trainer = Item2VecTrainer(self, self.model)
        self.model = trainer.train(dataloader, data_repr=self.data_repr)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from scripts.dataset import get_datasets 
    
    # Configuration
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
    # Pick a heavy user to see multiple sessions
    valid_users = user_counts[user_counts > 50].index
    
    selected_user_id = valid_users[0]
    df_user = df_real[df_real[kw.COLUMN_USER_ID] == selected_user_id].copy()

    # Model Init
    model = Item2vec_temp_model(
        embedding_dir="tmp", 
        time_exp=1.5, 
        min_time_diff=60 
    )

    df_processed = model.timestamp_diff(df_user)
    
    sessions = df_processed['increment'].values

    timestamps = df_processed[kw.COLUMN_TIME_DIFF].fillna(0).cumsum().values
    
    unique_sessions = np.unique(sessions)
    print(f"User split into {len(unique_sessions)} distinct sessions.")
    
    plt.figure(figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_sessions)))
    
    for i, sess_id in enumerate(unique_sessions):
        mask = sessions == sess_id
        plt.scatter(np.where(mask)[0], timestamps[mask], color=colors[i % 10], label=f'Session {sess_id}', s=50)
        
    plt.title(f'Temporal Session Splitting\nUser {selected_user_id}', fontsize=14)
    plt.xlabel('Item Sequence Index')
    plt.ylabel('Cumulative Time (Approx)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    print("Graph generated. Different colors represent distinct training sessions (Pseudo-Users).")