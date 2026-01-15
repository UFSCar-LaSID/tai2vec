import scripts as kw

ALS_HYPERPARAMETERS = {
    'factors': [50, 100, 200],
    'regularization': [1e-2, 1e-4, 1e-6],
    'iterations': [10, 20, 50, 100]
}

BPR_HYPERPARAMETERS = {
    'factors': [50, 100, 200],
    'learning_rate': [0.0025, 0.025, 0.25],
    'regularization': [1e-2, 1e-4, 1e-6],
    'iterations': [10, 20, 50, 100]
}

GEMSIM_HYPERPARAMETERS = {
    'learning_rate': [0.01, 0.0025, 0.00025],
    'sample': [0.01, 0.001, 0.0001, 0.00001],
    'negative_sampling': [5],
    'ns_exp':[-0.75, -0.5, 0.5, 0.75],
    'epochs': [10, 20, 40],
}

ITEMSIM_RECOMMENDER_HYPERPARAMETERS = {
    'recomender_norm': [True],
    'combination_strategy': ['avg_norm_before']
} #'target_only', 'avg_norm_after', 'avg_norm_before'

ITEM2VEC_HYPERPARAMETERS = {
    'w_size': [5, 10],
    'learning_rate': [0.25, 0.025],
    'subsample': [1e-3, 1e-4, 1e-5],
    'negative_samples': [7],
    'negative_exp': [-1, -0.5, 0.5, 1],
    'regularization': [1e-6],
    'epochs': [20, 50, 100],
}

ITEM2VEC_TEMP_HYPERPARAMETERS = {
    'w_size': [10],
    'learning_rate': [0.25, 0.025],
    'subsample': [1e-3, 1e-4, 1e-5],
    'negative_samples': [7],
    'negative_exp': [-1, -0.5, 0.5, 1],
    'regularization': [1e-6],
    'epochs': [20, 50, 100],
    'time_exp': [1, 1.5, 2],
    'min_time_diff': [300],
}

ITEM2VEC_CONT_HYPERPARAMETERS = {
    'w_size': [10],
    'learning_rate': [0.25, 0.025],
    'subsample': [1e-3, 1e-4, 1e-5],
    'negative_samples': [7],
    'negative_exp': [-1, -0.5, 0.5, 1],
    'regularization': [1e-6],
    'epochs': [20, 50, 100],
    'weight_floor': [0.2, 0.5],
    'decay_rate': [3, 5],
    'min_time_diff': [300],
}

ITEM2VEC_CONT_EXP_HYPERPARAMETERS = {
    #'factors': [50],
    'w_size': [20],
    'learning_rate': [0.25, 0.025, 0.0025],
    'subsample': [0.01],
    'negative_samples': [7],
    'negative_exp': [-1, -0.5, 0.5, 1],
    #'regularization': [-1],
    'batch_size': [2**12],
    'epochs': [20, 80],
    'lr_decay': [0.0001],
    'decay_rate': [0.01, 0.1],
    'weight_floor': [0.3],
}


ALS_ITEM_SIM_HYPERPARAMETERS = {

}

BPR_ITEM_SIM_HYPERPARAMETERS = {

}