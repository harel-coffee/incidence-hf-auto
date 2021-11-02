import argparse
import operator
from functools import partial
from typing import List

import numpy as np
import optuna
from mlflow import get_experiment_by_name, start_run, set_tracking_uri, set_tag, log_params

from deps.common import get_variables
# noinspection PyUnresolvedReferences
from deps.ignore_warnings import *
from deps.logger import logger
from deps.methods import METHODS_DEFINITIONS
from hcve_lib.custom_types import FoldPrediction
from hcve_lib.cv import Optimize, predict_survival
from hcve_lib.cv import lco_cv
from hcve_lib.evaluation_functions import c_index
from hcve_lib.tracking import log_pickled, log_metrics_ci


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


def test_callback(trial):
    print("TRIAL")
    ...


def run_lco_optimized(methods: List[str]) -> None:
    data, metadata, X, y = get_variables()
    logger.info('Data loaded')
    cv = lco_cv(data.groupby('STUDY'))
    set_tracking_uri('http://localhost:5000')
    optuna.logging.set_verbosity(optuna.logging.INFO)

    for method_name in methods:
        method_definition = METHODS_DEFINITIONS[method_name]
        experiment = get_experiment_by_name('lco_optimized')
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
                            'n_trials': 50,
                        },
                        optimize_callbacks=[
                            test_callback,
                            EarlyStoppingCallback(early_stopping_rounds=20, direction='maximize')
                        ],
                    )
                    pipeline.fit(
                        X,
                        method_definition['process_y'](y),
                    )
                    log_params(pipeline.study.best_trial.user_attrs)
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
