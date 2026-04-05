
import sys
import os

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(parent_path)

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = "1"
import pandas as pd
import scripts as kw
import numpy as np
from scripts.modules.dataset import get_datasets 

from scripts.modules.utils.file_handlers import get_embeddings_filepath, get_recomendation_filepath, get_metrics_filepath, log_recommendations, get_all_embeddings_filepath, str_to_dict
from scripts.modules.recommenders import get_recommenders
from scripts.modules.metrics import Metrics
from scripts.modules.utils.parameters_handle import get_input
from scripts.modules.dataset import DATASETS_TABLE, remove_single_interactions, remove_cold_start
from scripts.modules.recommenders import RECOMMENDERS_TABLE
from shutil import rmtree
import torch

torch.manual_seed(kw.RANDOM_STATE)
np.random.seed(kw.RANDOM_STATE)

print("Using PyTorch version:", torch.__version__, "with CUDA support:", torch.cuda.is_available())

dataset_options, recommender_options = get_input('Choose algorithmns and recommenders options to test', [
    {
        'name': 'datasets',
        'description': 'Dataset names (or indexes) to use. If not provided, a interactive menu will be shown. If "all" is provided, all datasets will be preprocessed.',
        'options': DATASETS_TABLE,
        'name_column': kw.DATASET_NAME
    },
    {
        'name': 'recommenders',
        'description': 'Recommender names (or indexes) to use. If not provided, a interactive menu will be shown. If "all" is provided, all recommenders will be used.',
        'options': RECOMMENDERS_TABLE,
        'name_column': kw.RECOMMENDER_NAME
    }
])

dataset_names = [DATASETS_TABLE.loc[option_index, kw.DATASET_NAME] for option_index in dataset_options]
recommender_names = [RECOMMENDERS_TABLE.loc[option_index, kw.RECOMMENDER_NAME] for option_index in recommender_options]

MODES = ['TrainEmbeddings', 'Recommend', 'Evaluate']     

# 'Recommend', 'Evaluate', 'TrainEmbeddings'

def train_embeddings(df, embeddings_filepath, embedding_model, parameters):
    Embedding_model = embedding_model(embeddings_filepath, **parameters)
    Embedding_model.fit(df)
    return Embedding_model

def recommend(df_train, df_test, embeddings_filepath, recomendation_filepath, recommender_model, parameters):

    if isinstance(parameters, dict):
        rec_param = parameters
    else:
        rec_param = str_to_dict(parameters)
    
    model = recommender_model(embeddings_filepath=embeddings_filepath, use_norm=rec_param['recomender_norm'], combination_strategy=rec_param['combination_strategy'])
    model.fit(df_train)
    recommendations = model.recommend(df_test)
    return log_recommendations(recomendation_filepath, parameters, df_test, recommendations)
    
def evaluate(recomendation_filepath, metrics_filepath):
    metrics_model = Metrics(kw.N_EVAL)
    metrics_model.add_metrics(recomendation_filepath)
    metrics_model.save_metrics(metrics_filepath)    
    return metrics_model.get_best_parameters(kw.EVALUATION_PARAMETER)

for dataset in get_datasets(datasets=dataset_names):

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
    df_train, df_remaining = train_test_split(df, test_size=0.3, shuffle=False)
    df_val_aux, df_test = train_test_split(df_remaining, test_size=0.5, shuffle=False)
    df_val = remove_cold_start(df_train, df_val_aux)

    if ('TrainEmbeddings' in MODES):
        
        for recommender in get_recommenders(recommenders=recommender_names):

            recommender_name = recommender.get_name()

            print('Training Embeddings - Dataset: {} | Recommender: {}'.format(dataset_name, recommender_name))                                
            
            # Itera por todas as embeddings 
            for parameters in tqdm(ParameterGrid(recommender.get_embeddings_hyperparameters())):   

                #Define onde salvar as embeddings
                embeddings_filepath = get_embeddings_filepath(kw.VALIDATION, dataset_name, recommender.get_embeddings_name(), parameters)
                embedding_model = train_embeddings(df_train, embeddings_filepath, recommender.get_embeddings_model(), parameters)

                # Como ALS e BPR fazem treino e recomendação em um passo, eles são tratados aqui.
                if recommender_name == 'ALS' or recommender_name == 'BPR':
                    recommendations = embedding_model.recommend(df_val)
                    recomendation_filepath = get_recomendation_filepath(kw.VALIDATION, dataset_name, recommender_name)
                    log_recommendations(recomendation_filepath, parameters, df_val, recommendations)

    if ('Recommend' in MODES): 

        for recommender in get_recommenders(recommenders=recommender_names):

            recommender_name = recommender.get_name()

            if recommender_name == 'ALS' or recommender_name == 'BPR':
                continue

            embedding_paths, embedding_params_list = get_all_embeddings_filepath(kw.VALIDATION, dataset_name, recommender.get_embeddings_name())
            recomendation_filepath = get_recomendation_filepath(kw.VALIDATION, dataset_name, recommender_name)

            print('Recommendations - Dataset: {} | Recommender: {}'.format(dataset_name, recommender_name))

            for i in tqdm(range(len(embedding_paths))):
                embedding_path = embedding_paths[i]
                embedding_params = str_to_dict(embedding_params_list[i])

                for rec_params in ParameterGrid(recommender.get_recommender_hyperparameters()):
                    
                    combined_params = {**embedding_params, **rec_params}
                    recommend(df_train, df_val, embedding_path, recomendation_filepath, recommender.get_model(), combined_params)

    if ('Evaluate' in MODES):

        #Atualiza o treino concatenando a validação a ele, e remove os usuários de cold start do teste
        df_train = pd.concat([df_train, df_val_aux], axis=0)
        df_test = remove_cold_start(df_train, df_test)

        for recommender in get_recommenders(recommenders=recommender_names):

            recommender_name = recommender.get_name()
            embedding_name = recommender.get_embeddings_name()

            recomendation_filepath = get_recomendation_filepath(kw.VALIDATION, dataset_name, recommender_name)
            metrics_filepath = get_metrics_filepath(kw.VALIDATION, dataset_name, recommender_name)

            print('Evaluate - Dataset: {} | Recommender: {}'.format(dataset_name, recommender_name))

            best_parameters = evaluate(recomendation_filepath, metrics_filepath)
            print('Best parameters: {}'.format(best_parameters))

            #-----------------------------------//-----------------------------------

            best_parameters_dict = str_to_dict(best_parameters)
            
            # Separa os parâmetros de embedding e de recomendação
            embedding_params = {k: v for k, v in best_parameters_dict.items() if k in recommender.get_embeddings_hyperparameters()}
            rec_params = {k: v for k, v in best_parameters_dict.items() if k in recommender.get_recommender_hyperparameters()}

            embeddings_filepath = get_embeddings_filepath(kw.TEST, dataset_name, recommender.get_embeddings_name(), embedding_params)
            recomendation_filepath = get_recomendation_filepath(kw.TEST, dataset_name, recommender_name)
            metrics_filepath = get_metrics_filepath(kw.TEST, dataset_name, recommender_name)

            # Treina as embeddings com os melhores parâmetros no dataset de treino+validação
            embedding_model = train_embeddings(df_train, embeddings_filepath, recommender.get_embeddings_model(), embedding_params)

            # A partir do melhor parâmetro, realiza a recomendação para os dados de teste
            if recommender_name == 'ALS' or recommender_name == 'BPR':
                recommendations = embedding_model.recommend(df_test)
                log_recommendations(recomendation_filepath, best_parameters, df_test, recommendations)
            else:
                # Combina os parâmetros novamente para a função de recomendação
                recommend(df_train, df_test, embeddings_filepath, recomendation_filepath, recommender.get_model(), best_parameters_dict)

            evaluate(recomendation_filepath, metrics_filepath)

    if 'Evaluate' in MODES:
        rmtree(os.path.join('results', 'recommendations', kw.VALIDATION, dataset_name))
