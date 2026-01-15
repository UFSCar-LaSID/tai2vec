import numpy as np
import pickle
import os
import scripts as kw
from scripts.recommenders.Item2vec.Data_repr import DataRepr
from .utils.recommendations import get_recommendations
import pandas as pd
import faiss
import torch

def get_topk_cosine_faiss(embeddings, k):

    x = embeddings.astype(np.float32)
    n, d = x.shape
    x = np.ascontiguousarray(x)

    index_cpu = faiss.IndexFlatIP(d)

    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    else:
        index = index_cpu

    index.add(x)

    scores, neighbors = index.search(x, k + 1)

    return neighbors[:, 1:], scores[:, 1:]

def combine_embeddings(target_embeddings, context_embeddings, combination_strategy='avg_norm_after', use_norm=True):

    t = torch.from_numpy(target_embeddings).float()
    c = torch.from_numpy(context_embeddings).float()

    def _norm(x):
        return x / (x.norm(dim=1, keepdim=True) + 1e-9)

    if use_norm:
        t = _norm(t)
        c = _norm(c)
        
    combined = t + c

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

        n_items = self.item_embeddings.shape[0]
        self.k = min(self.k, n_items - 1)

        top_k_indices, top_k_scores = get_topk_cosine_faiss(
            self.item_embeddings,
            self.k
        )

        item_ids = self.data_repr.get_item_id(np.arange(n_items))

        self.item_item_sim = pd.DataFrame({
            kw.COLUMN_ITEM_ID: np.repeat(item_ids, self.k),
            'neighbor': self.data_repr.get_item_id(top_k_indices.flatten()),
            'sim': top_k_scores.flatten()
        })

    def recommend(self, df_test):
        return get_recommendations(self.df_train, df_test, self.item_item_sim, self.data_repr)