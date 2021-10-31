from typing import Dict, Tuple

from optuna import Trial


def gb_optuna(trial: Trial) -> Tuple[Trial, Dict]:
    hyperparameters = {
        'estimator': {
            'learning_rate': trial.suggest_uniform('estimator_learning_rate', 0, 1),
            'max_depth': trial.suggest_int('estimator_max_depth', 1, 10),
            'n_estimators': trial.suggest_int('estimator_n_estimators', 5, 200),
            'min_samples_split': trial.suggest_int('estimator_min_samples_split', 2, 30),
            'min_samples_leaf': trial.suggest_int('estimator_min_samples_leaf', 1, 200),
            'max_features': trial.suggest_categorical(
                'estimator_max_features', ['auto', 'sqrt', 'log2']
            ),
            'subsample': trial.suggest_uniform('estimator_subsample', 0.1, 1),
        }
    }
    return trial, hyperparameters


def coxnet_optuna(trial: Trial) -> Tuple[Trial, Dict]:
    hyperparameters = {
        'estimator': {
            'l1_ratio': 1 - trial.suggest_loguniform('estimator_n_alphas', 0.1, 1),
        }
    }
    return trial, hyperparameters


def rsf_optuna(trial: Trial) -> Tuple[Trial, Dict]:
    hyperparameters = {
        'estimator': {
            'n_estimators': trial.suggest_int('estimator_n_estimators', 5, 200),
            'max_depth': trial.suggest_int('estimator_max_depth', 1, 30),
            'min_samples_split': trial.suggest_int('estimator_min_samples_split', 2, 30),
            'min_samples_leaf': trial.suggest_int('estimator_min_samples_leaf', 1, 200),
            'max_features': trial.suggest_categorical(
                'estimator_max_features', ['auto', 'sqrt', 'log2']
            ),
            'oob_score': trial.suggest_categorical('estimator_oob_score', [True, False]),
        }
    }
    return trial, hyperparameters
