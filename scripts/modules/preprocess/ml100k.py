# Link para download da base original: https://grouplens.org/datasets/movielens/

import scripts as kw
import os
import pandas as pd
from datetime import datetime


def preprocess_ml100k(input_path: str, output_path: str):
    '''
    Preprocesses the Movielens-100k dataset.

    params:
        input_path: Path to the input directory containing the dataset files.
        output_path: Path to the output directory where the processed files will be saved.
    '''

    os.makedirs(output_path, exist_ok=True)

    df_interactions = pd.read_csv(os.path.join(input_path, 'u.data'), sep='\t', encoding='latin-1', header=None)
    df_interactions.columns = [kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID, kw.COLUMN_RATING, kw.COLUMN_TIMESTAMP]
    df_interactions[kw.COLUMN_DATETIME] = df_interactions[kw.COLUMN_TIMESTAMP].apply(lambda x: datetime.fromtimestamp(x))
    df_interactions = df_interactions[[kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID, kw.COLUMN_RATING, kw.COLUMN_DATETIME, kw.COLUMN_TIMESTAMP]]
    df_interactions = df_interactions.sort_values([kw.COLUMN_DATETIME, kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID])

    for df, file_name in [(df_interactions, kw.FILE_INTERACTIONS)]:
        df.to_csv(os.path.join(output_path, file_name), sep=kw.DELIMITER, encoding=kw.ENCODING, quoting=kw.QUOTING, quotechar=kw.QUOTECHAR, header=True, index=False)