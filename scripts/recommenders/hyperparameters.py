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

GEMSIM_HYPERPARAMETERS = {
    'learning_rate': [0.25], #0.1, 0.25, 
    'sample': [0.001],
    #'negative_samples': [3, 10],
    'ns_exp': [-0.75],
    #'batch_size': [2**16],
    'epochs': [50, 100], #150, 200
}

ITEM2VEC_HYPERPARAMETERS = {
    #'factors': [128],
    'w_size': [20],
    'learning_rate': [0.25], #0.1, 0.25, 
    'min_learning_rate': [0.025, 0.0025],
    'subsample': [0.001, 0.0001],
    #'negative_samples': [3],
    'negative_exp': [-0.75],
    'lr_decay': [0.2, 0.5],
}


ITEM2VEC_TEMP_HYPERPARAMETERS = {
    #'factors': [128],
    'w_size': [-1],
    'learning_rate': [0.025], #0.1, 0.25, 
    'min_learning_rate': [0.025, 0.0025],
    'subsample': [0.01],
    #'negative_samples': [3],
    'negative_exp': [0.5, 0.75, 1],
    'lr_decay': [0.3, 0.5],
    'time_exp': [0, 1, 1.5],
    'min_time_diff': [300, 86500]
}

ITEM2VEC_CONT_HYPERPARAMETERS = {
    #'factors': [128],
    'w_size': [20],
    'learning_rate': [0.025], #0.1, 0.25, 
    'min_learning_rate': [0.025, 0.0025],
    'subsample': [0.01],
    #'negative_samples': [3],
    'negative_exp': [0.5, 0.75, 1],
    'lr_decay': [0.2, 0.5],
    'curve_exp': [-1],
    'min_weight': [0.1, 0.3],
    'weight_floor': [0.1, 0.3],
    'min_time_diff': [300, 86500]
}

ALS_ITEM_SIM_HYPERPARAMETERS = {

}

BPR_ITEM_SIM_HYPERPARAMETERS = {

}