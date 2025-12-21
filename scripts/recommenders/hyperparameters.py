import scripts as kw

ALS_HYPERPARAMETERS = {
    'factors': [50, 100],
    'regularization': [1e-2, 1e-4, 1e-6],
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

ITEMSIM_RECOMMENDER_HYPERPARAMETERS = {
    'recomender_norm': [True, False],
    'combination_strategy': ['avg_norm_after', 'avg_norm_before', 'target_only']
} #'target_only', 'avg_norm_after', 'avg_norm_before'

ITEM2VEC_HYPERPARAMETERS = {
    #'factors': [50],
    'w_size': [5, 10],
    'learning_rate': [0.25, 0.0025],
    'subsample': [0.01, 0.001],
    #'negative_samples': [7],
    'negative_exp': [-1, -0.5, 0.5, 1],
    'regularization': [-1],
    #'batch_size': [2**12],
    'epochs': [5, 20, 50, 100],
    'big_innit': [False],
    #'lr_decay': [0.0001],
}

ITEM2VEC_TEMP_HYPERPARAMETERS = {
    #'factors': [50],
    'w_size': [5, 10],
    'learning_rate': [0.25, 0.0025],
    'subsample': [0.01, 0.001],
    #'negative_samples': [7],
    'negative_exp': [-1, -0.5, 0.5, 1],
    'regularization': [-1],
    #'batch_size': [2**12],
    'epochs': [5, 20, 50, 100],
    'big_innit': [False],
    'min_time_diff': [300],
}

ITEM2VEC_CONT_HYPERPARAMETERS = {
    #'factors': [50],
    'w_size': [5, 10],
    'learning_rate': [0.25, 0.0025],
    'subsample': [0.01, 0.001],
    #'negative_samples': [7],
    'negative_exp': [-1, -0.5, 0.5, 1],
    'regularization': [-1],
    #'batch_size': [2**12],
    'epochs': [5, 20, 50, 100],
    'big_innit': [False],
    'weight_floor': [0.3],
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