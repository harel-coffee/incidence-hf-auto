import argparse
import operator
from functools import partial
from logging import INFO
from typing import List

import mlflow
import numpy as np
import optuna
from mlflow import get_experiment_by_name, start_run, set_tracking_uri, set_tag, log_params
from optuna.trial import TrialState

from deps.common import get_variables
# noinspection PyUnresolvedReferences
from deps.ignore_warnings import *
from deps.logger import logger
from deps.methods import METHODS_DEFINITIONS
from hcve_lib.custom_types import FoldPrediction
from hcve_lib.cv import Optimize
from hcve_lib.cv import lco_cv
from hcve_lib.evaluation_functions import c_index
from hcve_lib.tracking import log_pickled, log_metrics_ci, log_text


class EarlyStoppingCallback(object):
    """Early stopping callback for Optuna."""

    def __init__(self, early_stopping_rounds: int, direction: str = "minimize") -> None:
        self.early_stopping_rounds = early_stopping_rounds

        self._iter = 0

        if direction == "minimize":
            self._operator = operator.lt
            self._score = np.inf
        elif direction == "maximize":
            self._operator = operator.gt
            self._score = -np.inf
        else:
            ValueError(f"invalid direction: {direction}")

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        print('iter', self._iter)
        """Do early stopping."""
        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1

        if self._iter >= self.early_stopping_rounds:
            study.stop()


direction = "maximize"
study = optuna.create_study(direction=direction)


def mlflow_runs(study, _):
    set_tag(
        'trials',
        len(study.trials),
    )
    mlflow.log_figure(
        optuna.visualization.plot_optimization_history(study),
        'optimization_history.html',
    )
    mlflow.log_figure(
        optuna.visualization.plot_parallel_coordinate(study),
        'parallel_coordinate_hyperparams.html',
    )
    if len(study.get_trials(states=[TrialState.COMPLETE])) > 1:
        mlflow.log_figure(
            optuna.visualization.plot_param_importances(study),
            'plot_hyperparam_importances.html',
        )
    return study


def run_lco_optimized(methods: List[str]) -> None:
    data, metadata, X, y = get_variables()
    logger.setLevel(INFO)
    logger.info('Data loaded')
    cv = lco_cv(data.groupby('STUDY'))
    set_tracking_uri('http://localhost:5000')
    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    experiment = get_experiment_by_name('lco_optimized')
    for method_name in methods:
        method_definition = METHODS_DEFINITIONS[method_name]
        # TODO: fix
        # mlflow_callback = MLflowCallback(nest_trials=True)
        with start_run(run_name=method_name, experiment_id=experiment.experiment_id):
            for fold_name, fold in cv.items():
                with start_run(
                    run_name=fold_name,
                    experiment_id=experiment.experiment_id,
                    nested=True,
                ):
                    pipeline = Optimize(
                        partial(method_definition['get_estimator'], verbose=0),
                        method_definition['optuna'],
                        scoring,
                        method_definition['predict'],
                        [fold],
                        optimize_params={
                            'n_jobs': -1,
                            'n_trials': 500,
                        },
                        optimize_callbacks=[
                            mlflow_runs,
                            EarlyStoppingCallback(early_stopping_rounds=20, direction='maximize')
                        ],
                        logger=logger,
                    )
                    pipeline.fit(
                        X,
                        method_definition['process_y'](y),
                    )
                    log_text(pipeline.study.best_trial.user_attrs['pipeline'], 'pipeline.txt')
                    mlflow.log_param(
                        'hyperparameters', pipeline.study.best_trial.user_attrs['hyperparameters']
                    )
                    log_metrics_ci(pipeline.study.best_trial.user_attrs['metrics'])
                    set_tag("method_name", method_name)
                    log_pickled(pipeline.study, 'study')


def scoring(estimator, X_test, y_true):
    y_score = estimator.predict(X_test)
    return c_index(FoldPrediction(y_true=y_true, y_score=y_score, model=estimator))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('methods', metavar='METHOD', type=str, nargs='+')
    args = parser.parse_args()
    run_lco_optimized(**vars(args))
