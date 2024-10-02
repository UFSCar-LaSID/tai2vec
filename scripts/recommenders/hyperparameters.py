import scripts as kw

ALS_HYPERPARAMETERS = {
    'factors': [32, 64, 128],
    'regularization': [0.001, 0.01, 0.1],
    'iterations': [15, 30, 50]
}

BPR_HYPERPARAMETERS = {
    'factors': [32, 64, 128],
    'learning_rate': [0.001, 0.01, 0.1],
    'regularization': [0.001, 0.01, 0.1],
    'iterations': [50, 100, 200]
}

ITEM2VEC_HYPERPARAMETERS = {
    #'factors': [128],
    'w_size': [3, 5, 7],
    'learning_rate': [0.25], #0.1, 0.25, 
    'subsample': [0.001, 0.0001],
    #'negative_samples': [3],
    'negative_exp': [-1, -0.75, 0.75, 1],
    #'batch_size': [2**16],
    'epochs': [120], #150, 200
}

GEMSIM_HYPERPARAMETERS = {
    'learning_rate': [0.25, 0.025], #0.1, 0.25, 
    'sample': [0.001, 0.01],
    #'negative_samples': [3, 10],
    'ns_exp': [-1, -0.5, 0.75, 1],
    #'batch_size': [2**16],
    'epochs': [50, 100], #150, 200
}

ITEM2VEC_TEMP_HYPERPARAMETERS = {
    #'factors': [128],
    #'w_size': [2, 3, 5, 7],
    'learning_rate': [0.25], #0.1, 0.25, 
    'subsample': [0.001],
    #'negative_samples': [3],
    'negative_exp': [-0.5],
    #'batch_size': [2**16],
    #'epochs': [100], #150, 200
    'time_exp': [0, 1, 2]
}

ALS_ITEM_SIM_HYPERPARAMETERS = {

}

BPR_ITEM_SIM_HYPERPARAMETERS = {

}

ITEM2VEC_CONT_HYPERPARAMETERS = {
    #'factors': [128],
    #'w_size': [-1],
    'learning_rate': [0.25], #0.1, 0.25, 
    'subsample': [0.0001, 0.00001],
    #'negative_samples': [3, 10],
    'negative_exp': [-1, -0.5, 0.75, 1],
    #'batch_size': [2**16],
    'epochs': [100, 150], #150, 200
}