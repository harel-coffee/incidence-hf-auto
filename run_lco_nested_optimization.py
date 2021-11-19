import argparse
from functools import partial
from logging import INFO
from typing import List, Hashable, Dict

import optuna
from mlflow import get_experiment_by_name, start_run, set_tracking_uri, set_tag, log_metrics
from pandas import DataFrame

from common import brier
from deps.common import get_variables_cached
# noinspection PyUnresolvedReferences
from deps.custom_types import MethodDefinition
from deps.logger import logger
from hcve_lib.custom_types import Target, FoldPrediction
from hcve_lib.cv import cross_validate, filter_missing_features, OptimizeEstimator, evaluate_optimize_splits, \
    train_test_proportion
from hcve_lib.cv import get_lco_splits
from hcve_lib.evaluation_functions import c_index, compute_metrics_folds
from hcve_lib.optimization import optuna_report_mlflow, EarlyStoppingCallback
from hcve_lib.utils import partial2
from pipelines import get_pipelines

study = optuna.create_study(direction="maximize")


def run(methods: List[str]) -> None:
    data, metadata, X, y = get_variables_cached()
    logger.setLevel(INFO)
    logger.info('Data loaded')
    cv = get_lco_splits(X, y, data)
    set_tracking_uri('http://localhost:5000')
    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    experiment = get_experiment_by_name('lco_nested')
    for method_name in methods:
        method_definition = get_pipelines()[method_name]
        with start_run(run_name=method_name, experiment_id=experiment.experiment_id):
            set_tag("method_name", method_name)
            set_tag("nested", True)
            result: Dict[Hashable, FoldPrediction] = cross_validate(
                X,
                y,
                lambda _: get_inner_optimize(data, method_definition=method_definition),
                method_definition.predict,
                cv,
                train_test_filter_callback=filter_missing_features,
                n_jobs=-1,
                mlflow_track=True,
            )

            report_final(X, y, result, experiment.experiment_id)


def get_inner_optimize(data: DataFrame, method_definition: MethodDefinition):
    return OptimizeEstimator(
        method_definition.get_estimator,
        method_definition.predict,
        method_definition.optuna,
        objective_evaluate=partial(
            evaluate_optimize_splits,
            predict_callback=method_definition.predict,
            get_cv=train_test_proportion,
        ),
        get_cv=partial(get_lco_cv, data=data),
        optimize_params={
            'n_jobs': -1,
            'n_trials': 50,
        },
        optimize_callbacks=[
            optuna_report_mlflow,
            EarlyStoppingCallback(
                early_stopping_rounds=30,
                direction='maximize',
                stop_callback=log_early_stopping,
            )
        ],
        catch_exceptions=True,
        logger=logger,
    )


def report_final(
    X: DataFrame, y: Target, result: Dict[Hashable, FoldPrediction], experiment_id: str
):
    metrics_folds = compute_metrics_folds(
        result,
        [
            partial2(c_index, kwargs={
                'X': X,
                'y': y
            }),
            partial2(
                brier,
                kwargs={
                    'X': X,
                    'y': y,
                    'time_point': 3 * 365,
                },
                name='brier_3_years',
            ),
        ],
    )
    for fold_name, fold_metrics in metrics_folds.items():
        with start_run(run_name=fold_name, nested=True, experiment_id=experiment_id):
            log_metrics(fold_metrics)


def get_lco_cv(X, y, data):
    splits = get_lco_splits(X, y, data)
    return splits


def log_early_stopping(it):
    logger.info(f'Early stopping after {it} iterations')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('methods', metavar='METHOD', type=str, nargs='+')
    args = parser.parse_args()
    run(**vars(args))
