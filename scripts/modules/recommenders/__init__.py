import pandas as pd

import scripts as kw
from scripts.modules.recommenders.Mf_models.mf import ALS, BPR
from scripts.modules.recommenders.hyperparameters import ALS_HYPERPARAMETERS, BPR_HYPERPARAMETERS, ITEMSIM_RECOMMENDER_HYPERPARAMETERS, ITEM2VEC_HYPERPARAMETERS, ITEM2VEC_TEMP_HYPERPARAMETERS, ITEM2VEC_CONT_HYPERPARAMETERS
from scripts.modules.recommenders.itemSim import ItemSim
from scripts.modules.recommenders.Item2vec.Item2vec_base import Item2vec_model
from scripts.modules.recommenders.Item2vec.Item2vec_disc import Item2vec_temp_model
from scripts.modules.recommenders.Item2vec.Item2Vec_disc_aug import Item2vec_temp_aug_model
from scripts.modules.recommenders.Item2vec.Item2vec_cont import Item2vec_Temp_Cont_model

RECOMMENDERS_TABLE = pd.DataFrame(
    [[1,  'ALS',                  "ALS",                  ALS,                        ALS,          ALS_HYPERPARAMETERS,               {}],
     [2,  'BPR',                  "BPR",                  BPR,                        BPR,          BPR_HYPERPARAMETERS,               {}],
     [3,  'Item2Vec_itemSim',     "Item2Vec_itemSim",     Item2vec_model,             ItemSim,      ITEM2VEC_HYPERPARAMETERS,          ITEMSIM_RECOMMENDER_HYPERPARAMETERS],
     [4,  'TimeI2V_Disc_Aug',     "TimeI2V_Disc_Aug",     Item2vec_temp_aug_model,    ItemSim,      ITEM2VEC_TEMP_HYPERPARAMETERS,     ITEMSIM_RECOMMENDER_HYPERPARAMETERS],
     [5,  'TimeI2V_Cont',         "TimeI2V_Cont",         Item2vec_Temp_Cont_model,   ItemSim,      ITEM2VEC_CONT_HYPERPARAMETERS,     ITEMSIM_RECOMMENDER_HYPERPARAMETERS],
     [6,  'Seq2Vec',              "Seq2Vec",              Item2vec_Temp_Cont_model,   ItemSim,      ITEM2VEC_CONT_HYPERPARAMETERS,     ITEMSIM_RECOMMENDER_HYPERPARAMETERS]],
    columns=[kw.RECOMMENDER_ID, kw.RECOMMENDER_NAME, kw.EMBEDDING_NAME, kw.RECOMMENDER_EMBEDDINGS, kw.RECOMMENDER_CLASS, kw.EMBEDDINGS_HYPERPARAMETERS, kw.RECOMMENDER_HYPERPARAMETERS]
).set_index(kw.RECOMMENDER_ID)

class Recommender(object):
    def __init__(self, recommender_id):
        self.name = RECOMMENDERS_TABLE.loc[recommender_id, kw.RECOMMENDER_NAME]
        self.embeddings_model = RECOMMENDERS_TABLE.loc[recommender_id, kw.RECOMMENDER_EMBEDDINGS]
        self.embeddings_name = RECOMMENDERS_TABLE.loc[recommender_id, kw.RECOMMENDER_NAME]
        self.model = RECOMMENDERS_TABLE.loc[recommender_id, kw.RECOMMENDER_CLASS]
        self.rec_hyperparameters = RECOMMENDERS_TABLE.loc[recommender_id, kw.RECOMMENDER_HYPERPARAMETERS]
        self.emb_hyperparameters = RECOMMENDERS_TABLE.loc[recommender_id, kw.EMBEDDINGS_HYPERPARAMETERS]

    def get_name(self):
        return self.name
    
    def get_embeddings_name(self):
        return self.embeddings_name
    
    def get_embeddings_model(self):
        return self.embeddings_model
    
    def get_model(self):
        return self.model
    
    def get_recommender_hyperparameters(self):
        return self.rec_hyperparameters

    def get_embeddings_hyperparameters(self):
        return self.emb_hyperparameters
    
    def get_all_hyperparameters(self):
        return {
            **self.rec_hyperparameters,
            **self.emb_hyperparameters
        }
    
    def get_embeddings_hyperparameter_from_dict(self, all_hyperparameters_dict):
        emb_hyperparameters_dict = {}
        for key, value in all_hyperparameters_dict.items():
            if key in self.emb_hyperparameters:
                emb_hyperparameters_dict[key] = value
        return emb_hyperparameters_dict

def get_recommenders(recommenders=None):
    for recommender_id, recommender_data in RECOMMENDERS_TABLE.iterrows():
        if recommenders is None or recommender_data[kw.RECOMMENDER_NAME] in recommenders:
            yield Recommender(recommender_id)