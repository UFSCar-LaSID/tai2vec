import scripts as kw
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import cupy as cp
import time
import torch

class ItemSim(object):
    def __init__(self, embeddings_filepath, k=kw.K, **model_params):
        self.k = k
        self.sparse_repr = pickle.load(open(os.path.join(embeddings_filepath, kw.FILE_SPARSE_REPR), 'rb'))
        self.embeddings = np.load(open(os.path.join(embeddings_filepath, kw.FILE_ITEMS_EMBEDDINGS), 'rb'), allow_pickle=True)
        #self.embeddings = self.embeddings / np.sqrt(np.sum(self.embeddings**2, axis=1)).reshape(-1,1)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-8)

    def fit(self, df):
        print("[ItemSim.fit] Iniciando o cálculo de similaridade entre itens...")
        start_time = time.time()

        self.df_train = df
        n_items = self.embeddings.shape[0]
        items_per_batch = int(2**6)

        embeddings_gpu = torch.tensor(self.embeddings, dtype=torch.float32).cuda()
        sim_data = []

        for i in range(0, n_items, items_per_batch):

            batch_start = i
            batch_end = min(i + items_per_batch, n_items)

            batch_items = torch.arange(batch_start, batch_end, device='cuda')
            batch_embs = embeddings_gpu[batch_start:batch_end]
            batch_sims = torch.matmul(batch_embs, embeddings_gpu.t())

            # Set diagonal to -inf for the batch
            diag_indices = torch.arange(batch_end - batch_start, device='cuda')
            batch_sims[diag_indices, batch_start + diag_indices] = float('-inf')

            rows = batch_items.repeat_interleave(self.k)
            top_k_vals, top_k_idx = torch.topk(batch_sims, self.k, dim=1)
            sim_data.append(
                torch.stack([rows, top_k_idx.flatten(), top_k_vals.flatten()], dim=1).cpu().numpy()
            )

        sim_matrix_cpu = np.vstack(sim_data)

        self.item_item_sim = pd.DataFrame(sim_matrix_cpu, columns=[kw.COLUMN_ITEM_ID, 'neighbor', 'sim'])

        elapsed_time = time.time() - start_time
        print(f"[ItemSim.fit] Tempo decorrido: {elapsed_time:.2f} segundos")

    def recommend(self, df_test):
                
        target_users = df_test[kw.COLUMN_USER_ID].unique()        
        item_based_neighborhood = pd.merge(self.df_train[self.df_train[kw.COLUMN_USER_ID].isin(target_users)], self.item_item_sim, on=kw.COLUMN_ITEM_ID, how='inner')
        final_sim = item_based_neighborhood.groupby([kw.COLUMN_USER_ID, 'neighbor'])['sim'].mean().reset_index()
        
        final_sim = final_sim.merge(
            self.df_train, 
            how='left', 
            left_on=[kw.COLUMN_USER_ID, 'neighbor'], 
            right_on=[kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID]
        )

        final_sim = final_sim[final_sim[kw.COLUMN_ITEM_ID].isna()].drop(columns=[kw.COLUMN_ITEM_ID])
        recommendations = final_sim.sort_values('sim', ascending=False).groupby(kw.COLUMN_USER_ID).head(kw.TOP_N).sort_values([kw.COLUMN_USER_ID, 'sim'], ascending=[True, False])
        del final_sim
        
        recommendations[kw.COLUMN_RANK] = np.concatenate(recommendations.groupby(kw.COLUMN_USER_ID).size().sort_index(ascending=True).apply(lambda x:np.arange(1, x+1)).values)
        recommendations = recommendations.rename(columns={'neighbor': kw.COLUMN_ITEM_ID})[[kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID, "rank"]].reset_index(drop=True)

        return recommendations