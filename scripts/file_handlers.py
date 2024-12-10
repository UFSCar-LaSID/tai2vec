import os
import pandas as pd

import scripts as kw

MAIN_FOLDER = 'results'

def _dict_to_str(dictionary):
    return '@'.join(['{}={}'.format(k,v) for k, v in sorted(dictionary.items())])

def str_to_dict(string):
    def convert_value(value):
        return int(value) if value.isdigit() else float(value)

    return {key: convert_value(value) for key, value in (item.split('=') for item in string.split('@'))}


def get_embeddings_filepath(file_type, dataset_name, recommender_name, parameters):
    if isinstance(parameters, dict):
        parameters = _dict_to_str(parameters)
    filepath = os.path.join(MAIN_FOLDER, 'embeddings', file_type, dataset_name, recommender_name, parameters)
    return filepath

def get_recomendation_filepath(file_type, dataset_name, recommender_name):
    filepath = os.path.join(MAIN_FOLDER, 'recommendations', file_type, dataset_name, recommender_name)
    return filepath

def get_metrics_filepath(file_type, dataset_name, recommender_name):
    filepath = os.path.join(MAIN_FOLDER, 'metrics', file_type, dataset_name, recommender_name)
    return filepath

def get_all_embeddings_filepath(file_type, dataset_name, recommender_name, embeddings_parameters='all'):

    files_path = []
    parameters_string = []
    
    main_path = os.path.join(MAIN_FOLDER, 'embeddings', file_type, dataset_name, recommender_name)
    for curr_file in os.listdir(main_path):
        files_path.append(os.path.join(main_path, curr_file))
        parameters_string.append(curr_file)

    return files_path, parameters_string

def log_recommendations(recomendation_filepath, parameters_string, df_test, recommendations):
    if isinstance(parameters_string, dict):
        parameters_string = _dict_to_str(parameters_string)
    filedir = os.path.join(recomendation_filepath, parameters_string)
    os.makedirs(filedir, exist_ok=True)
    filepath = os.path.join(filedir, 'recommendations.csv')
    user_items = df_test.groupby(kw.COLUMN_USER_ID)[kw.COLUMN_ITEM_ID].apply(lambda x: list(x))
    user_recs = recommendations.groupby(kw.COLUMN_USER_ID)[kw.COLUMN_ITEM_ID].apply(lambda x: list(x))
    recommendations_match = pd.concat([user_items, user_recs], axis=1).reset_index()
    recommendations_match.columns = [kw.LOG_COLUMN_USER, kw.LOG_COLUMN_ITEMS, kw.LOG_COLUMN_RECOMMENDATIONS]
    recommendations_match.to_csv(filepath, sep=kw.DELIMITER, header=True, index=False, encoding=kw.ENCODING, quoting=kw.QUOTING, quotechar=kw.QUOTECHAR)
    return filedir

def log_items_similarity(file_path, items_similarities):
    items_similarities.to_csv(f'{file_path}/similarities.csv', sep=kw.DELIMITER, header=True, index=False, encoding=kw.ENCODING, quoting=kw.QUOTING, quotechar=kw.QUOTECHAR)