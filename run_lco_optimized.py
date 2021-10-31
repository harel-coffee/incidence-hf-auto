import argparse
import operator
from typing import List

import numpy as np
import optuna
from mlflow import get_experiment_by_name, start_run, set_tracking_uri, set_tag, log_metric, log_params

from deps.common import get_variables_cached, METHODS_DEFINITIONS
# noinspection PyUnresolvedReferences
from deps.ignore_warnings import *
from deps.pipelines import get_pipeline
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
early_stopping = EarlyStoppingCallback(10, direction=direction)


def run_lco_optimized(methods: List[str]) -> None:
    data, metadata, X, y = get_variables_cached()

    cv = lco_cv(data.groupby('STUDY'))

    set_tracking_uri('http://localhost:5000')

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
                        lambda: get_pipeline(method_definition['get_estimator'](), X),
                        method_definition['optuna'],
                        scoring,
                        predict_survival,
                        [fold],
                        optimize_params={
                            'n_jobs': -1,
                            'n_trials': 1000
                        },
                        optimize_callbacks=[
                            EarlyStoppingCallback(early_stopping_rounds=100, direction='maximize')
                        ],
                    )
                    pipeline.fit(X, y)
                    log_params(pipeline.study.best_trial.user_attrs)
                    log_metrics_ci(pipeline.study.best_trial.user_attrs['metrics'])
                    set_tag("method_name", method_name)
                    log_pickled(pipeline.study, 'study')

        #
        # #
        # for estimator_name, estimator in methods.items():
        #     with start_run(run_name=estimator_name, experiment_id=experiment.experiment_id):
        #         result = run_prediction(
        #             cv,
        #             data,
        #             metadata,
        #             lambda: get_pipeline(estimator, X),
        #         )
        # See the tracking docs for a list of
        #         log_pickled(result, 'result')
        #         metrics_ci = compute_metrics_ci(result['predictions'], [c_index])
        #         log_metrics_ci(metrics_ci)
        #
        #         metrics_folds = compute_metrics_folds(result['predictions'], [c_index])
        #         for fold_name, fold_metrics in metrics_folds.items():
        #             with start_run(run_name=fold_name, nested=True, experiment_id=experiment.experiment_id):
        #                 log_metrics(fold_metrics)

        # for name, metrics_fold in metrics_folds.items():
        #     with start_method_run(name):
        #


def scoring(estimator, X_test, y_true):
    y_score = estimator.predict(X_test)
    return c_index(FoldPrediction(y_true=y_true, y_score=y_score, model=estimator))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('methods', metavar='METHOD', type=str, nargs='+')
    args = parser.parse_args()
    run_lco_optimized(**vars(args))
