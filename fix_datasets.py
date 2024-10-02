import csv
import os
import pandas as pd

ROOT_FOLDER = 'datasets'
FINAL_FILE = 'interactions.csv'

for dataset_folder in os.listdir(ROOT_FOLDER):
    print(dataset_folder)
    
    explicit_file = os.path.join(ROOT_FOLDER, dataset_folder, 'interactions_explicit.csv')
    implicit_file = os.path.join(ROOT_FOLDER, dataset_folder, 'interactions_implicit.csv')
    recurrent_file = os.path.join(ROOT_FOLDER, dataset_folder, 'interactions_recurrent.csv')
    final_file = os.path.join(ROOT_FOLDER, dataset_folder, FINAL_FILE)
    
    if dataset_folder == 'Book-Crossing':
        df = pd.read_csv(implicit_file, header=0, sep=';', quotechar='"', quoting=csv.QUOTE_ALL, encoding="ISO-8859-1")
        df['interaction'] = df['interaction'].replace({0:-1})
        df.to_csv(final_file, header=True, sep=';', quotechar='"', quoting=csv.QUOTE_ALL, encoding="ISO-8859-1")
        os.remove(explicit_file)
        os.remove(implicit_file)

    elif os.path.exists(explicit_file):        
        if os.path.exists(implicit_file):
            os.remove(implicit_file)
        os.rename(explicit_file, final_file)

    elif os.path.exists(implicit_file):
        os.rename(implicit_file, final_file)

    if os.path.exists(recurrent_file):
        os.remove(recurrent_file)
