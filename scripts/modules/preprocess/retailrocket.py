
# Link para download da base original: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
from datetime import datetime
import os
import pandas as pd
import scripts as kw

def preprocess_retailrocket():
    # Pasta de dataset
    DATASET_FOLDER = 'RetailRocket'

    # Nomes das colunas
    COLUMN_INTERACTION_TYPE = 'type'
    COLUMN_TRANSACTION_ID = 'id_transaction'
    COLUMN_DATETIME = 'datetime'
    COLUMN_TIMESTAMP = 'timestamp'

    # Define pasta de leitura
    input_dir = os.path.join(kw.RAW_FOLDER, DATASET_FOLDER)

    # ========================================= INTERAÇÕES =========================================
    # Lê CSV de interações
    df_interactions = pd.read_csv(os.path.join(input_dir, 'events.csv'), sep=',', header=0, index_col=False)

    # Arruma nome das colunas
    df_interactions.columns = ['wrong_timestamp', kw.COLUMN_USER_ID, COLUMN_INTERACTION_TYPE, kw.COLUMN_ITEM_ID, COLUMN_TRANSACTION_ID]

    # Padroniza timestamps de itens da mesma interação
    last_timestamps = df_interactions.groupby(COLUMN_TRANSACTION_ID)['wrong_timestamp'].max()
    df_interactions[COLUMN_TIMESTAMP] = df_interactions[COLUMN_TRANSACTION_ID].map(last_timestamps.to_dict()).fillna(df_interactions['wrong_timestamp']).astype(int)
    df_interactions.drop(columns='wrong_timestamp')
    df_interactions = df_interactions.drop_duplicates(keep='first')

    # Gera datetime
    df_interactions[COLUMN_DATETIME] = df_interactions[COLUMN_TIMESTAMP].apply(lambda x: str(datetime.fromtimestamp(x // 1000)))

    # Reordena colunas
    df_interactions = df_interactions[[COLUMN_DATETIME, kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID, COLUMN_INTERACTION_TYPE]]


    # ============================ GERAR ARQUIVOS APENAS COM TRANSACOES ============================
    df_interactions = df_interactions[df_interactions[COLUMN_INTERACTION_TYPE]=='transaction']
    df_interactions = df_interactions.drop(columns=[COLUMN_INTERACTION_TYPE])

    # =========================================== SALVAR ===========================================
    # Cria pasta de saída
    output_dir = os.path.join(kw.PREPROCESSED_DATASET_FOLDER, '{}-transactions'.format(DATASET_FOLDER))
    os.makedirs(output_dir, exist_ok=True)

    # Salva bases
    df_interactions.to_csv(os.path.join(output_dir, kw.FILE_INTERACTIONS), sep=kw.DELIMITER, encoding=kw.ENCODING, quoting=kw.QUOTING, quotechar=kw.QUOTECHAR, header=True, index=False)