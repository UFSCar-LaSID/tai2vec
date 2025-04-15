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
    'learning_rate': [0.025, 0.0025, 0.00025],
    'sample': [0.01, 0.001, 0.0001, 0.00001],
    'negative_sampling': [5],
    'ns_exp':[-0.75, -0.5, 0.5, 0.75],
    'epochs': [10, 20, 40],
}

ITEM2VEC_HYPERPARAMETERS = {
    'w_size': [-1],
    'learning_rate': [0.01, 0.001],
    'subsample': [0.01, 0.001],
    'negative_exp': [-1, -0.5, 0.5, 1],
    'lr_decay': [0.96],
    'regularization': [1e-5, 1e-6],
}


ITEM2VEC_TEMP_HYPERPARAMETERS = {
    'w_size': [-1],
    'learning_rate': [0.0025, 0.001],
    'subsample': [0.01, 0.001],
    'negative_exp': [-1, -0.5, 0.5, 1],
    'lr_decay': [0.96],
    'regularization': [1e-5, 1e-6],
    'time_exp': [1, 1.5, 2],
    'min_time_diff': [86300]
}

ITEM2VEC_CONT_HYPERPARAMETERS = {
    'factors': [100],
    'w_size': [-1, 5],
    'learning_rate': [0.25, 0.025, 0.0025],
    'subsample': [0.001, 0.0001],
    'negative_samples': [5, 10],
    'negative_exp': [-0.75, -0.5, 0.5, 0.75],
    'curve_exp': [-1, 2],
    'min_weight': [0.1],
    'weight_floor': [0.3],
    'min_time_diff': [300]
}

ALS_ITEM_SIM_HYPERPARAMETERS = {

}

BPR_ITEM_SIM_HYPERPARAMETERS = {

}