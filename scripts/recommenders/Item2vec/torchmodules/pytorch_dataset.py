import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class Item2VecDataset(Dataset):
    
    def __init__(self, X_target, X_context, cumulative_table, negative_samples, weights=None):
        self.X_target = np.asarray(X_target, dtype=np.int32)
        self.X_context = np.asarray(X_context, dtype=np.int32)
        self.cumulative_table_cpu = torch.tensor(cumulative_table, dtype=torch.float32) 
        self.negative_samples = negative_samples
        
        if weights is None:
            self.weights = np.ones(len(X_target), dtype=np.float32)
        else:
            self.weights = np.asarray(weights, dtype=np.float32)
    
    def __len__(self):
        return len(self.X_target)
    
    def __getitem__(self, idx):
        return np.array([self.X_target[idx], self.X_context[idx], self.weights[idx]], dtype=np.float32)

    def negative_sampling_collate(self, batch):
        
        batch_np = np.stack(batch, axis=0)  # shape [B, 3]
        target_items = torch.from_numpy(batch_np[:, 0].astype(np.int32))
        positive_contexts = torch.from_numpy(batch_np[:, 1].astype(np.int32))
        weights = torch.from_numpy(batch_np[:, 2].astype(np.float32))   # [B]

        batch_size = target_items.size(0)

        # Expande os itens-alvo de [batch_size] para [batch_size, negative_samples+1]
        targets = target_items.unsqueeze(1).expand(-1, self.negative_samples + 1).contiguous()

        # Coleta os contextos negativos
        random_samples = torch.rand(batch_size, self.negative_samples)  # CPU
        negative_contexts = torch.searchsorted(self.cumulative_table_cpu, random_samples, right=True)

        contexts = torch.cat([positive_contexts.unsqueeze(1), negative_contexts], dim=1)

        labels = torch.cat([
            torch.ones(batch_size, 1, dtype=torch.float32),
            torch.zeros(batch_size, self.negative_samples, dtype=torch.float32)
        ], dim=1)

        weights_full = torch.cat([
            weights.unsqueeze(1),
            weights.unsqueeze(1).expand(-1, self.negative_samples)
        ], dim=1)

        return targets, contexts, labels, weights_full

def create_item2vec_dataloader(X_target, X_context, cumulative_table, negative_samples, batch_size=1024, weights=None, shuffle=True, num_workers=0):
    
    dataset = Item2VecDataset(X_target, X_context, cumulative_table, negative_samples, weights)
    
    # enable pin_memory when CUDA is available so trainer's .to(..., non_blocking=True) is effective
    pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers= num_workers > 0,
        collate_fn=dataset.negative_sampling_collate,
        prefetch_factor=None
    )