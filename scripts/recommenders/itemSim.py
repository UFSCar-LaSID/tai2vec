import scripts as kw
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ItemSim(object):
    def __init__(self, embeddings_filepath, k=kw.K, **model_params):
        self.k = k
        self.sparse_repr = pickle.load(open(os.path.join(embeddings_filepath, kw.FILE_SPARSE_REPR), 'rb'))
        self.embeddings = np.load(open(os.path.join(embeddings_filepath, kw.FILE_ITEMS_EMBEDDINGS), 'rb'), allow_pickle=True)
        self.embeddings = self.embeddings / np.sqrt(np.sum(self.embeddings**2, axis=1)).reshape(-1,1)

    def fit(self, df):
        
        n_items = self.embeddings.shape[0]
        
        items_per_batch = int(kw.MEM_SIZE_LIMIT / (8 * n_items))
        #if items_per_batch == 0:
        #    items_per_batch = kw.MEM_SIZE_LIMIT

        self.item_item_sim = pd.DataFrame()        
        for i in range(0, n_items, items_per_batch):
            batch_items = self.sparse_repr.get_item_id(np.arange(i, min(i+items_per_batch, n_items)))
            batch_sims = np.dot(self.embeddings[i:i+items_per_batch], self.embeddings.T) # calcula similaridade            
            np.fill_diagonal(batch_sims[:, i:i+items_per_batch], -np.inf)
            self.item_item_sim = pd.concat([
                self.item_item_sim,    
                pd.DataFrame(
                    np.column_stack([
                        np.repeat(batch_items, self.k),
                        self.sparse_repr.get_item_id(np.argpartition(-batch_sims, kth=self.k-1, axis=1)[:, :self.k].flatten()), # captura os itens vizinhos
                        -np.partition(-batch_sims, kth=self.k-1, axis=1)[:, :self.k].flatten() # captura similaridades dos k vizinhos
                    ]),
                    columns=[kw.COLUMN_ITEM_ID, 'neighbor', 'sim']
                )
            ])
        self.df_train = df.copy()

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
        recommendations = recommendations.astype({
            kw.COLUMN_ITEM_ID: 'int64',
            kw.COLUMN_USER_ID: 'int64'
        })

        return recommendations