from configs.main_config import seed

histgbm_params = {
    "loss": "log_loss",
    "learning_rate": 0.01,
    "max_iter": 1800,
    "max_depth": 5,
    "max_features": 0.8,
    "early_stopping": False,
    "validation_fraction": None,
    "verbose": 0,
    "random_state": seed,
    "categorical_features": "from_dtype",
    "class_weight": "balanced",
}

lightgbm_params = {
    "boosting_type": "goss", # gradient based one side sampling
    "metric": "auc",
    "n_estimators": 2500,
    "learning_rate": 0.01,
    "num_leaves": 31,
    "max_depth": -1,
    # "subsample": 0.8, # cannot be used with goss
    "colsample_bytree": 0.7,
    "random_state": seed,
    "n_jobs": -1,
    "importance_type": "gain",
    "verbose": -1,
    "is_unbalance": True,
}

xgboost_params = {
    "n_estimators": 2000,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "learning_rate": 0.01,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": seed,
    "n_jobs": -1,
    "device": "cuda",
    "verbosity":0,
}

catboost_params = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 1200,
    "learning_rate": 0.01,
    "depth": 5,
    "l2_leaf_reg": 3.0,
    "random_seed": seed,
    "early_stopping_rounds": 150,
    "thread_count": -1,
    "verbose": False,
    "bootstrap_type": "Bayesian",
    "task_type": "GPU",
    "auto_class_weights": "Balanced",
    "metric_period": 5,
}