import numpy as np
import pickle
import os
import scripts as kw
from scripts.recommenders.Item2vec.Data_repr import DataRepr
from .utils.recommendations import get_recommendations
import pandas as pd
import torch

def get_cosine_similarity_matrix(embeddings, use_norm=True, batch_size=128):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    emb = torch.from_numpy(embeddings).to(device=device, dtype=torch.float32)

    n = emb.shape[0]
    sim_matrix = torch.empty((n, n), device=device, dtype=torch.float32)

    # Process in blocks: rows x cols
    bs = int(batch_size)
    for i in range(0, n, bs):
        a = emb[i:i+bs]  # [bs, d]
        # Optionally further split columns for very large n
        for j in range(0, n, bs):
            b = emb[j:j+bs]  # [bs, d] (or smaller on last block)
            # [bs_i, d] @ [d, bs_j] -> [bs_i, bs_j]
            block = a @ b.T
            sim_matrix[i:i+a.shape[0], j:j+b.shape[0]] = block

        del a
        torch.cuda.empty_cache() if device == 'cuda' else None

    return sim_matrix.detach().cpu().numpy()

def combine_embeddings(target_embeddings, context_embeddings, combination_strategy='avg_norm_after', use_norm=True):

    t = torch.from_numpy(target_embeddings).float()
    c = torch.from_numpy(context_embeddings).float()

    def _norm(x):
        return x / (x.norm(dim=1, keepdim=True) + 1e-9)

    if combination_strategy == 'avg_norm_before':
        if use_norm:
            t = _norm(t)
            c = _norm(c)
        combined = (t + c) / 2.0

    elif combination_strategy == 'avg_norm_after':
        combined = (t + c) / 2.0
        if use_norm:
            combined = _norm(combined)

    elif combination_strategy == 'target_only':
        combined = t
        if use_norm:
            combined = _norm(combined)

    else:
        raise ValueError(f"Unknown combination strategy: {combination_strategy}")

    return combined.numpy()

class ItemSim:

    def __init__(self, embeddings_filepath, use_norm=True, combination_strategy='avg_norm_after', k=kw.K):
        
        self.embedding_dir = embeddings_filepath
        self.use_norm = use_norm
        self.combination_strategy = combination_strategy
        self.k = k

        with open(os.path.join(embeddings_filepath, kw.FILE_SPARSE_REPR), 'rb') as f:
            self.data_repr = pickle.load(f)
        
        target_embeddings = np.load(os.path.join(embeddings_filepath, kw.FILE_ITEMS_EMBEDDINGS))
        context_embeddings = np.load(os.path.join(embeddings_filepath, kw.FILE_CONTEXT_EMBEDDINGS))

        self.item_embeddings = combine_embeddings(
            target_embeddings,
            context_embeddings,
            combination_strategy=self.combination_strategy,
            use_norm=self.use_norm
        )

    def fit(self, df):

        self.df_train = df

        sim_matrix = get_cosine_similarity_matrix(self.item_embeddings, self.use_norm)

        n_items = sim_matrix.shape[0]
        self.k = min(self.k, n_items - 1)

        np.fill_diagonal(sim_matrix, -np.inf)

        top_k_indices = np.argpartition(-sim_matrix, kth=self.k-1, axis=1)[:, :self.k]
        top_k_scores = np.array([sim_matrix[i, top_k_indices[i]] for i in range(n_items)])

        item_ids = self.data_repr.get_item_id(np.arange(n_items))
        
        self.item_item_sim = pd.DataFrame({
            kw.COLUMN_ITEM_ID: np.repeat(item_ids, self.k),
            'neighbor': self.data_repr.get_item_id(top_k_indices.flatten()),
            'sim': top_k_scores.flatten()
        })

    def recommend(self, df_test):
        return get_recommendations(self.df_train, df_test, self.item_item_sim, self.data_repr)