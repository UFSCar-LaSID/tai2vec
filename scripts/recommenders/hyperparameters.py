import scripts as kw

ALS_HYPERPARAMETERS = {
    'factors': [50, 100],
    'regularization': [1e-2, 1e-4],
    'iterations': [25, 50, 100]
}

BPR_HYPERPARAMETERS = {
    'factors': [50, 100],
    'learning_rate': [0.0025, 0.025, 0.25],
    'regularization': [1e-2, 1e-4],
    'iterations': [25, 50, 100]
}

GEMSIM_HYPERPARAMETERS = {
    'learning_rate': [0.01, 0.0025, 0.00025],
    'sample': [0.01, 0.001, 0.0001, 0.00001],
    'negative_sampling': [5],
    'ns_exp':[-0.75, -0.5, 0.5, 0.75],
    'epochs': [10, 20, 40],
}

ITEM2VEC_HYPERPARAMETERS = {
    'factors': [50],
    'w_size': [10],
    'learning_rate': [0.25, 0.025, 0.0025],
    'subsample': [0.01, 0.001],
    'negative_samples': [7],
    'negative_exp': [-1, -0.5, 0.5, 1],
    'regularization': [-1],
    'batch_size': [2**16],
    'epochs': [20, 50, 100],
    'lr_decay': [1e-6],
    'recomender_norm': ['False'],
}

ITEM2VEC_TEMP_HYPERPARAMETERS = {
    'factors': [50],
    'w_size': [10],
    'learning_rate': [0.25, 0.025, 0.0025],
    'subsample': [0.01, 0.001],
    'negative_samples': [7],
    'negative_exp': [-1, -0.5, 0.5, 1],
    'regularization': [-1],
    'batch_size': [2**16],
    'epochs': [20, 80],
    'lr_decay': [1e-6],
    'recomender_norm': ['False'],
    'time_exp': [1.5],
    'min_time_diff': [300],
}

ITEM2VEC_CONT_HYPERPARAMETERS = {
    'factors': [50],
    'w_size': [10],
    'learning_rate': [0.25, 0.025, 0.0025],
    'subsample': [0.01, 0.001],
    'negative_samples': [7],
    'negative_exp': [-1, -0.5, 0.5, 1],
    'regularization': [-1],
    'batch_size': [2**16],
    'epochs': [20, 80],
    'lr_decay': [1e-6],
    'recomender_norm': ['False'],
    'min_time_diff': [300],
    'min_weight': [0.2],
    'curve_exp': [-1, 2],
    'weight_floor': [0.2],
}

ALS_ITEM_SIM_HYPERPARAMETERS = {

}

BPR_ITEM_SIM_HYPERPARAMETERS = {

}