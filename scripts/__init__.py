import csv
import numpy as np

# Dados gerais do experimento
RANDOM_STATE = 2002
K_FOLD_SPLITS = 5
TRAIN_MODE = 'cpu'

# Nomes de arquivos da base de dados
FILE_ITEMS = 'items.csv'
FILE_USERS = 'users.csv'
FILE_INTERACTIONS = 'interactions.csv'

# Nomes de arquivos de saves de modelos
FILE_SPARSE_REPR = 'sparse_repr.pkl'
FILE_ITEMS_EMBEDDINGS = 'items.npy'
FILE_USERS_EMBEDDINGS = 'users.npy'

# Colunas da tabela de dataset
DATASET_ID = 'id'
DATASET_NAME = 'name'
DATASET_TYPE = 'type'
DATASET_SAMPLING_RATE = 'sampling_rate'

# Colunas da tabela de recomendadores
RECOMMENDER_ID = 'id'
RECOMMENDER_NAME = 'name'
RECOMMENDER_EMBEDDINGS = 'embeddings'
EMBEDDING_NAME = 'embedding_name'
RECOMMENDER_CLASS = 'class'
RECOMMENDER_HYPERPARAMETERS = 'specific_recommender_hyperparameter'
EMBEDDINGS_HYPERPARAMETERS = 'embeddings_hyperparameters'

# Dados dos CSVs
DELIMITER = ';'
QUOTECHAR = '"'
QUOTING = csv.QUOTE_ALL
ENCODING = "ISO-8859-1"

# Colunas dos CSVs e dataframmes
COLUMN_ITEM_ID = 'id_item'
COLUMN_ITEM_NAME = 'name_item'
COLUMN_USER_ID = 'id_user'
COLUMN_USER_NAME = 'name_user'
COLUMN_RATING = 'rating'
COLUMN_RANK = 'rank'

# Colunas do arquivo de log
LOG_COLUMN_USER = 'user'
LOG_COLUMN_ITEMS = 'items'
LOG_COLUMN_RECOMMENDATIONS = 'recommendations'

# Colunas para a implementação temporal
COLUMN_TIMESTAMP = 'timestamp'
COLUMN_TIME_CUMSUM = 'timestamp_cumsum'
COLUMN_TIME_CUMSUM_NORM = 'timestamp_cumsum_norm'
COLUMN_DATETIME = 'datetime'
COLUMN_TIME_DIFF = 'timestamp_diff' 
COLUMN_MEAN = 'mean'
COLUMN_STD = 'std'
COLUMN_THRESHOLD = 'threshold'

#Tipo de recomendação
VALIDATION = 'validation'
TEST = 'test'

# Infos da recomendação
TOP_N = 25 # Não faço ideia se está sendo usado e tenho medo de remover

K = 100

# Ate que top-N deve calcular as metricas
N_EVAL = 20

# Limite de memória
MEM_SIZE_LIMIT = int(2**22)

