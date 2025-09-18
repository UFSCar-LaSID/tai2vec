import scripts as kw

ALS_HYPERPARAMETERS = {
    'factors': [50, 100],
    'regularization': [1e-2, 1e-4, 1e-6],
    'iterations': [25, 50, 100]
}

BPR_HYPERPARAMETERS = {
    'factors': [50, 100],
    'learning_rate': [0.0025, 0.025, 0.25],
    'regularization': [1e-2, 1e-4, 1e-6],
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
    'epochs': [100],
    'factors': [64],
    'w_size': [-1, 5],
    'learning_rate': [0.1, 0.01, 0.001],
    'subsample': [0.001],
    'negative_samples': [7],
    'batch_size': [2**14],
    'negative_exp': [1, 0.5, -0.5, -1],
    'regularization': [1e-3, 1e-4, 1e-5],
    'lr_decay': [0.96],
    'init_strat': ['uniform_small'],
    'recomender_norm': ['True'],
}

ITEM2VEC_TEMP_HYPERPARAMETERS = {
    'epochs': [100],
    'factors': [64],
    'w_size': [-1, 5],
    'learning_rate': [0.1, 0.01, 0.001],
    'subsample': [0.001],
    'negative_samples': [7],
    'batch_size': [2**14],
    'negative_exp': [1, 0.5, -0.5, -1],
    'regularization': [1e-3, 1e-4, 1e-5],
    'lr_decay': [0.96],
    'init_strat': ['uniform_small'],
    'recomender_norm': ['True'],
    'time_exp': [1, 2],
    'min_time_diff': [86300],
}

ITEM2VEC_CONT_HYPERPARAMETERS = {
    'epochs': [5],
    'factors': [64],
    'w_size': [20],
    'learning_rate': [0.001],
    'subsample': [0.001],
    'negative_samples': [5],
    'batch_size': [2**16],
    'negative_exp': [-1, -0.5, 0.5, 1],
    'regularization': [1e-5],
    'curve_exp': [-1],
    'min_weight': [0.1],
    'weight_floor': [0.3],
    'min_time_diff': [86300]
}

ALS_ITEM_SIM_HYPERPARAMETERS = {

}

BPR_ITEM_SIM_HYPERPARAMETERS = {

}