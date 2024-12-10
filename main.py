from sklearn.model_selection import KFold, ParameterGrid
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import scripts as kw
from scripts.dataset import get_datasets 


from scripts.file_handlers import get_embeddings_filepath, get_recomendation_filepath, get_metrics_filepath, log_recommendations, get_all_embeddings_filepath, str_to_dict
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

DATASETS = ['RetailRocket-Transactions']
#'RetailRocket-Transactions', 'DeliciousBookmarks', 'MovieLens', 'BestBuy',
#  'Taobao', 'Events', 'CiaoDVD', 'NetflixPrize'

RECOMMENDERS = ['TimeI2V_Disc_Aug']
# 'ALS', 'BPR'
# 'ALS_itemSim', 'BPR_itemSim',
# 'ALS_itemSim_temporal', 'BPR_itemSim_temporal', 
# 'Item2Vec_itemSim', 'TimeI2V_Disc', 'TimeI2V_Disc_Aug', 'TimeI2V_Cont'

MODES = ['Recommend', 'Evaluate']                                   
# 'Recommend', 'Evaluate'


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
    return metrics_model.get_best_parameters('NDCG@15')


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
    df_train, df_remaining = train_test_split(df, test_size=0.4, shuffle=False)
    df_val_aux, df_test = train_test_split(df_remaining, test_size=0.5, shuffle=False)
    df_val = remove_cold_start(df_train, df_val_aux)

    if ('Recommend' in MODES): 

        #Cria as embeddings de cada modelo
        for recommender in get_recommenders(recommenders=RECOMMENDERS):

            recommender_name = recommender.get_name()

            print('Embeddings - Dataset: {} | Recommender: {}'.format(dataset_name, recommender_name))                                            
            for parameters in tqdm(ParameterGrid(recommender.get_all_hyperparameters())):

                embeddings_filepath = get_embeddings_filepath(kw.VALIDATION, dataset_name, recommender.get_embeddings_name(), parameters)
                embedding_model = train_embeddings(df_train, embeddings_filepath, recommender.get_embeddings_model(), parameters)

                #Como o modelo de embeddings do ALS e BPR também fazem recomendações, eles são tratados separadamente
                if recommender_name == 'ALS' or recommender_name == 'BPR':
                    recommendations = embedding_model.recommend(df_val)
                    recomendation_filepath = get_recomendation_filepath(kw.VALIDATION, dataset_name, recommender_name)
                    rec_dir = log_recommendations(recomendation_filepath, parameters, df_val, recommendations)

        for recommender in get_recommenders(recommenders=RECOMMENDERS):

            recommender_name = recommender.get_name()

            #Ignora os modelos de embeddings ALS e BPR
            if recommender_name == 'ALS' or recommender_name == 'BPR':
                continue

            # Realiza as recomendações
            embeddings_filepath, parameters =  get_all_embeddings_filepath(kw.VALIDATION, dataset_name, recommender_name)
            recomendation_filepath = get_recomendation_filepath(kw.VALIDATION, dataset_name, recommender_name)

            print('Recommendations - Dataset: {} | Recommender: {}'.format(dataset_name, recommender_name))

            for i in tqdm(range(len(embeddings_filepath))):
                recommend(df_train, df_val, embeddings_filepath[i], recomendation_filepath, recommender.get_model(), parameters[i])

    if ('Evaluate' in MODES):

        #Atualiza o treino concatenando a validação a ele, e remove os usuários de cold start do teste
        df_train = pd.concat([df_train, df_val_aux], axis=0)
        df_test = remove_cold_start(df_train, df_test)

        for recommender in get_recommenders(recommenders=RECOMMENDERS):

            recommender_name = recommender.get_name()
            embedding_name = recommender.get_embeddings_name()

            recomendation_filepath = get_recomendation_filepath(kw.VALIDATION, dataset_name, recommender_name)
            metrics_filepath = get_metrics_filepath(kw.VALIDATION, dataset_name, recommender_name)

            print('Evaluate - Dataset: {} | Recommender: {}'.format(dataset_name, recommender_name))

            best_parameters = evaluate(recomendation_filepath, metrics_filepath)
            #-----------------------------------//-----------------------------------

            embeddings_filepath = get_embeddings_filepath(kw.TEST, dataset_name, recommender.get_embeddings_name(), best_parameters)
            recomendation_filepath = get_recomendation_filepath(kw.TEST, dataset_name, recommender_name)
            print('rec path:', recomendation_filepath)
            metrics_filepath = get_metrics_filepath(kw.TEST, dataset_name, recommender_name)

            embedding_model = train_embeddings(df_train, embeddings_filepath, recommender.get_embeddings_model(), str_to_dict(best_parameters))

            # A partir do melhor parâmetro, realiza a recomendação para os dados de teste
            if recommender_name == 'ALS' or recommender_name == 'BPR':
                recommendations = embedding_model.recommend(df_test)
                rec_dir = log_recommendations(recomendation_filepath, best_parameters, df_test, recommendations)
            else:
                recommend(df_train, df_test, embeddings_filepath, recomendation_filepath, recommender.get_model(), best_parameters)

            evaluate(recomendation_filepath, metrics_filepath) 