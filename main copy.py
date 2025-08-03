from sklearn.model_selection import KFold, ParameterGrid
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import scripts as kw
import numpy as np
from scripts.dataset import get_datasets 

from scripts.file_handlers import get_embeddings_filepath, get_recomendation_filepath, get_metrics_filepath, log_recommendations, get_all_embeddings_filepath, str_to_dict
from scripts.recommenders import get_recommenders
from scripts.recsys import remove_single_interactions, remove_cold_start
from scripts.metrics import Metrics
from shutil import rmtree

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    for i, gpu in enumerate(gpus):
        details = tf.config.experimental.get_device_details(gpu)
        print(f"GPU {i}: {details.get('device_name', gpu.name)}")
else:
    print('No GPU available')

DATASETS = ['CiaoDVD']
#'RetailRocket-Transactions', 'DeliciousBookmarks', 'MovieLens', 'BestBuy',
#'Taobao', 'Events', 'CiaoDVD', 'NetflixPrize', 'AmazonBooks', 'AmazonBeauty'

RECOMMENDERS = ['Item2Vec_itemSim']
# 'ALS', 'BPR'
# 'ALS_itemSim', 'BPR_itemSim',
# 'ALS_itemSim_temporal', 'BPR_itemSim_temporal', 
# 'Item2Vec_itemSim', 'TimeI2V_Disc', 'TimeI2V_Disc_Aug', 'TimeI2V_Cont', 'Gemsim_itemSim'

MODES = ['Recommend', 'Evaluate']                                   
# 'Recommend', 'Evaluate'

PARAMETER_TUNING = 'on_validation'
# 'on_test', 'on_validation'

def train_embeddings(df, embeddings_filepath, embedding_model, parameters):
    Embedding_model = embedding_model(embeddings_filepath, **parameters)
    Embedding_model.fit(df)
    return Embedding_model

def recommend(df_train, df_test, embeddings_filepath, recomendation_filepath, recommender_model, parameters):
    model = recommender_model(embeddings_filepath=embeddings_filepath)
    model.fit(df_train)
    recommendations = model.recommend(df_test)
    return log_recommendations(recomendation_filepath, parameters, df_test, recommendations)
    
def evaluate(recomendation_filepath, metrics_filepath):
    metrics_model = Metrics(kw.N_EVAL)
    metrics_model.add_metrics(recomendation_filepath)
    metrics_model.save_metrics(metrics_filepath)    
    return metrics_model.get_best_parameters(kw.EVALUATION_PARAMETER)

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
        
    #Ordena o dataframe pela coluna de tempo
    df = df.sort_values(by=kw.COLUMN_DATETIME)

    #Divide o dataset em treino, validação e teste
    if PARAMETER_TUNING == 'on_validation':
        df_train, df_remaining = train_test_split(df, test_size=0.2, shuffle=False)
        df_val_aux, df_test = train_test_split(df_remaining, test_size=0.5, shuffle=False)
        df_val = remove_cold_start(df_train, df_val_aux)
    elif PARAMETER_TUNING == 'on_test':
        df_train, df_val_aux = train_test_split(df, test_size=0.1, shuffle=False)
        df_val = remove_cold_start(df_train, df_val_aux)

    if ('Recommend' in MODES): 

        #Cria as embeddings de cada modelo
        for recommender in get_recommenders(recommenders=RECOMMENDERS):

            recommender_name = recommender.get_name()

            print('Embeddings - Dataset: {} | Recommender: {}'.format(dataset_name, recommender_name))                                            
            for parameters in tqdm(ParameterGrid(recommender.get_all_hyperparameters())):

                #Define onde salvar as embeddings
                embeddings_filepath = get_embeddings_filepath(kw.VALIDATION, dataset_name, recommender.get_embeddings_name(), parameters)
                embedding_model = train_embeddings(df_train, embeddings_filepath, recommender.get_embeddings_model(), parameters)

                #Como o modelo de embeddings do ALS e BPR também fazem recomendações, eles são tratados separadamente
                if recommender_name == 'ALS' or recommender_name == 'BPR':
                    recommendations = embedding_model.recommend(df_val)
                    recomendation_filepath = get_recomendation_filepath(kw.VALIDATION, dataset_name, recommender_name)
                    rec_dir = log_recommendations(recomendation_filepath, parameters, df_val, recommendations)

        for recommender in get_recommenders(recommenders=RECOMMENDERS):

            recommender_name = recommender.get_name()

            #Ignora os modelos de embeddings ALS e BPR, pois já fizarem a recomendação
            if recommender_name == 'ALS' or recommender_name == 'BPR':
                continue

            # Recebe onde as embeddings foram salvas e onde as recomendações devem ser encaminhadas
            embeddings_filepath, parameters = get_all_embeddings_filepath(kw.VALIDATION, dataset_name, recommender_name)
            recomendation_filepath = get_recomendation_filepath(kw.VALIDATION, dataset_name, recommender_name)

            print('Recommendations - Dataset: {} | Recommender: {}'.format(dataset_name, recommender_name))

            # Realiza as recomendações
            for i in tqdm(range(len(embeddings_filepath))):
                recommend(df_train, df_val, embeddings_filepath[i], recomendation_filepath, recommender.get_model(), parameters[i])

    if ('Evaluate' in MODES):

        #Atualiza o treino concatenando a validação a ele, e remove os usuários de cold start do teste
        if PARAMETER_TUNING == 'on_validation':
            df_train = pd.concat([df_train, df_val_aux], axis=0)
            df_test = remove_cold_start(df_train, df_test)

        for recommender in get_recommenders(recommenders=RECOMMENDERS):

            recommender_name = recommender.get_name()
            embedding_name = recommender.get_embeddings_name()

            recomendation_filepath = get_recomendation_filepath(kw.VALIDATION, dataset_name, recommender_name)
            metrics_filepath = get_metrics_filepath(kw.VALIDATION, dataset_name, recommender_name)

            print('Evaluate - Dataset: {} | Recommender: {}'.format(dataset_name, recommender_name))

            best_parameters = evaluate(recomendation_filepath, metrics_filepath)
            print('Best parameters: {}'.format(best_parameters))

            #-----------------------------------//-----------------------------------
            if PARAMETER_TUNING == 'on_validation':

                # Cria as embeddings de cada model
                embeddings_filepath = get_embeddings_filepath(kw.TEST, dataset_name, recommender.get_embeddings_name(), best_parameters)
                recomendation_filepath = get_recomendation_filepath(kw.TEST, dataset_name, recommender_name)
                metrics_filepath = get_metrics_filepath(kw.TEST, dataset_name, recommender_name)

                embedding_model = train_embeddings(df_train, embeddings_filepath, recommender.get_embeddings_model(), str_to_dict(best_parameters))

                # A partir do melhor parâmetro, realiza a recomendação para os dados de teste
                if recommender_name == 'ALS' or recommender_name == 'BPR':
                    recommendations = embedding_model.recommend(df_test)
                    rec_dir = log_recommendations(recomendation_filepath, best_parameters, df_test, recommendations)
                else:
                    embeddings_filepath, parameters = get_all_embeddings_filepath(kw.TEST, dataset_name, recommender_name)
                    for i in range(len(embeddings_filepath)):
                        recommend(df_train, df_test, embeddings_filepath[i], recomendation_filepath, recommender.get_model(), parameters[i])

                evaluate(recomendation_filepath, metrics_filepath)

    # Remove as pastas de validação
    rmtree(os.path.join('results', 'recommendations', kw.VALIDATION, dataset_name))


            

    