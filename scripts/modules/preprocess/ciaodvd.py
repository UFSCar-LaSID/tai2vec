import scripts as kw
import os
import pandas as pd

def preprocess_ciaodvd(input_path: str, output_path: str):
    '''
    Preprocesses the CiaoDVD dataset.

    params:
        input_path: Path to the input directory containing the dataset files.
        output_path: Path to the output directory where the processed files will be saved.
    '''

    os.makedirs(output_path, exist_ok=True)

    df_interactions = pd.read_csv(os.path.join(input_path, 'movie-ratings.txt'), header=None)
    df_interactions.columns = [kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID, 'genre_id', 'review_id', kw.COLUMN_RATING, kw.COLUMN_DATETIME]
    df_interactions[kw.COLUMN_DATETIME] = pd.to_datetime(df_interactions[kw.COLUMN_DATETIME], format='%Y-%m-%d')
    df_interactions = df_interactions[[kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID, kw.COLUMN_RATING, kw.COLUMN_DATETIME]]
    df_interactions = df_interactions[df_interactions[kw.COLUMN_RATING] >= 4]
    df_interactions = df_interactions[[kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID, kw.COLUMN_DATETIME]].sort_values([kw.COLUMN_DATETIME, kw.COLUMN_USER_ID])
    df_interactions = df_interactions.reset_index(drop=True)

    for df, file_name in [(df_interactions, kw.FILE_INTERACTIONS)]:
        df.to_csv(os.path.join(output_path, file_name), sep=kw.DELIMITER, encoding=kw.ENCODING, quoting=kw.QUOTING, quotechar=kw.QUOTECHAR, header=True, index=False)