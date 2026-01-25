# Link para download da base original: https://www.kaggle.com/competitions/acm-sf-chapter-hackathon-big

from datetime import datetime
import os
import pandas as pd
import scripts as kw

def preprocess_bestbuy(input_path: str, output_path: str):
    '''
    Preprocesses the Best Buy dataset.

    params:
        input_path: Path to the input directory containing the dataset files.
        output_path: Path to the output directory where the processed files will be saved.
    '''


    COLUMN_QUERY = 'query'
    COLUMN_QUERY_DATETIME = 'query_datetime'
    COLUMN_QUERY_TIMESTAMP = 'query_timestamp'

    os.makedirs(output_path, exist_ok=True)

    df_interactions = pd.read_csv(os.path.join(input_path, 'train.csv'), sep=',', header=0)

    df_interactions = df_interactions.drop(columns='category')
    df_interactions.columns = [kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID, COLUMN_QUERY, kw.COLUMN_DATETIME, COLUMN_QUERY_DATETIME]
    df_interactions[kw.COLUMN_DATETIME] = df_interactions[kw.COLUMN_DATETIME].apply(lambda x: datetime.strptime(x.split('.')[0], '%Y-%m-%d %H:%M:%S'))
    df_interactions[COLUMN_QUERY_DATETIME] = df_interactions[COLUMN_QUERY_DATETIME].apply(lambda x: datetime.strptime(x.split('.')[0], '%Y-%m-%d %H:%M:%S'))
    df_interactions[kw.COLUMN_TIMESTAMP] = df_interactions[kw.COLUMN_DATETIME].apply(lambda x: int(datetime.timestamp(x)))
    df_interactions[COLUMN_QUERY_TIMESTAMP] = df_interactions[COLUMN_QUERY_DATETIME].apply(lambda x: int(datetime.timestamp(x)))

    for df, file_name in [(df_interactions, kw.FILE_INTERACTIONS)]:
        df.to_csv(os.path.join(output_path, file_name), sep=kw.DELIMITER, encoding=kw.ENCODING, quoting=kw.QUOTING, quotechar=kw.QUOTECHAR, header=True, index=False)