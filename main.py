from sklearn.model_selection import KFold, ParameterGrid
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import scripts as kw
from scripts.dataset import get_datasets
from scripts.file_handlers import get_embeddings_filepath, get_recomendation_filepath, log_recommendations, get_all_embeddings_filepath
from scripts.recommenders import get_recommenders
from scripts.recsys import remove_single_interactions, remove_cold_start
from scripts.metrics import Metrics

import tensorflow as tf


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=6096)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)
else:
    print('No GPU available')

DATASETS = ['CiaoDVD']
#'RetailRocket-Transactions', 'DeliciousBookmarks', 'MovieLens', 'BestBuy', 'Taobao', 'Events'

RECOMMENDERS = ['TimeI2V_Cont'] 
# 'ALS', 'BPR'
# 'ALS_itemSim', 'BPR_itemSim',
# 'ALS_itemSim_temporal', 'BPR_itemSim_temporal', 
# 'Item2Vec_itemSim', 'Gemsim_itemSim', 'TimeI2V_Disc', 'TimeI2V_Cont'

MODES = ['Recommend', 'Evaluate']                                   
# 'Recommend', 'Evaluate'

for MODE in MODES:                                     

    if (MODE == 'Recommend'):

        for dataset in get_datasets(datasets=DATASETS):

            dataset_name = dataset.get_name()
            print('Loading dataset {}...'.format(dataset_name))
            
            df = dataset.get_dataframe()
            df = remove_single_interactions(df)

            if kw.COLUMN_TIMESTAMP in df.columns:
                df[kw.COLUMN_DATETIME] = pd.to_datetime(df[kw.COLUMN_TIMESTAMP], unit='s')
            elif kw.COLUMN_DATETIME in df.columns:
                df[kw.COLUMN_DATETIME] = pd.to_datetime(df[kw.COLUMN_DATETIME]).dt.floor('s')
            else:
                raise ValueError('Coluna temporal não encontrada')
                        
            df_train = df.sort_values(by=kw.COLUMN_DATETIME).iloc[:int(len(df) * 0.8)]
            df_test = df[~df.index.isin(df_train.index)].copy()
            df_test = remove_cold_start(df_train, df_test)
            
            for recommender in get_recommenders(recommenders=RECOMMENDERS):
                recommender_name = recommender.get_name()

                print('Embeddings - Dataset: {} | Recommender: {}'.format(dataset_name, recommender_name))                                            
                for parameters in tqdm(ParameterGrid(recommender.get_all_hyperparameters())):

                    embeddings_filepath = get_embeddings_filepath(
                        dataset_name, 
                        recommender.get_embeddings_name(), 
                        recommender.get_embeddings_hyperparameter_from_dict(parameters), 
                    )

                    #Treina e salva o modelo de Embeddings para todos os recomendadores
                    Embedding_model = recommender.get_embeddings_model()
                    embedding_model = Embedding_model(embeddings_filepath, **parameters)
                    embedding_model.fit(df_train)

                    if recommender_name == 'ALS' or recommender_name == 'BPR':
                        recommendations = embedding_model.recommend(df_test)
                        rec_dir = log_recommendations(dataset_name, recommender_name, parameters, df_test, recommendations)

            for recommender in get_recommenders(recommenders=RECOMMENDERS):
                recommender_name = recommender.get_name()

                #Não pergunte
                if recommender_name == 'ALS' or recommender_name == 'BPR':
                    continue

                # Realiza as recomendações
                embeddings_filepath, parameters =  get_all_embeddings_filepath(dataset_name, recommender_name)
                print('Recommendations - Dataset: {} | Recommender: {}'.format(dataset_name, recommender_name))
                for i in tqdm(range(len(embeddings_filepath))):
                    Model = recommender.get_model()
                    model = Model(embeddings_filepath=embeddings_filepath[i])
                    model.fit(df_train)
                    recommendations = model.recommend(df_test)

                    rec_dir = log_recommendations(dataset_name, recommender_name, parameters[i], df_test, recommendations)

    if (MODE == 'Evaluate'):

        for dataset in get_datasets(datasets=DATASETS):
            dataset_name = dataset.get_name()

            for recommender in get_recommenders(recommenders=RECOMMENDERS):
                recommender_name = recommender.get_name()

                print('Dataset: {} | Recommender: {}'.format(dataset_name, recommender_name))

                model = Metrics(kw.N_EVAL)
                recomendation_filepath = get_recomendation_filepath(dataset_name, recommender_name)
                print(recomendation_filepath)
                model.add_metrics(recomendation_filepath)
                model.save_metrics(dataset_name, recommender_name)
