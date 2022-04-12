from hcve_lib.log_output import capture_output, log_output
import argparse
from contextlib import contextmanager
from functools import partial
from logging import INFO, Logger
from typing import List, Type
from uuid import uuid4

import optuna
from mlflow import start_run, set_tracking_uri, set_tag, set_experiment
from pandas import DataFrame

from common import log_result
from deps.common import get_data
# noinspection PyUnresolvedReferences
from deps.constants import RANDOM_STATE, TRACKING_URI
from deps.logger import logger as global_logger
from deps.pipelines import get_pipelines
from hcve_lib.custom_types import Target, Method, Result
from hcve_lib.cv import OptimizeEstimator, evaluate_optimize_splits, repeated_cross_validate
from hcve_lib.evaluation_functions import c_index, compute_metrics_on_splits, make_brier
from hcve_lib.optimization import optuna_report_mlflow, EarlyStoppingCallback
from hcve_lib.splitting import filter_missing_features, get_splitter, get_train_test
from hcve_lib.tracking import update_nested_run, log_early_stopping, log_metric
from hcve_lib.utils import partial2, random_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('methods', metavar='METHOD', type=str, nargs='+')
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--n-repeats', type=int, default=1)
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--splits', type=str, dest='splits_name')
    args = parser.parse_args()
    run(**vars(args))


def run(methods: List[str], n_repeats: int, n_jobs: int, n_trials: int, splits_name: str) -> None:
    group_id = uuid4()
    random_seed(RANDOM_STATE)

    set_tracking_uri(TRACKING_URI)
    set_experiment(f'optimized_nested_{splits_name}')

    get_splits = get_splitter(splits_name)
    data, metadata, X, y = get_data()
    global_logger.setLevel(INFO)
    global_logger.info('Data loaded')
    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    for method_name in methods:
        method = get_pipelines()[method_name]
        results: List[Result] = repeated_cross_validate(
            get_pipeline=partial(
                get_nested_optimization,
                data=data,
                method=method,
                n_trials=n_trials,
                log=True,
                logger=global_logger,
            ),
            n_repeats=n_repeats,
            train_test_filter_callback=filter_missing_features,
            X=X,
            y=y,
            predict=method.predict,
            get_splits=partial(get_splits, data=data),
            get_repeat_mlflow_context=partial(get_context_for_repeat, method_name, group_id),
            on_repeat_result=partial(report_result, X, y, method, method_name),
            n_jobs=n_jobs,
            mlflow_track=True,
            random_state=RANDOM_STATE,
        )


def get_nested_optimization(
    _,
    random_state: int,
    data: DataFrame,
    method: Method,
    n_trials: int,
    log: bool = False,
    logger: Logger = None,
):
    return OptimizeEstimator(
        method.get_estimator,
        method.predict,
        method.optuna,
        objective_evaluate=partial(evaluate_optimize_splits),
        optimize_params={
            'n_jobs': 1,
            'n_trials': n_trials,
        },
        get_splits=partial2(get_train_test, randon_state=random_state, data=data),
        optimize_callbacks=[
            *([optuna_report_mlflow] if log else []),
            EarlyStoppingCallback(
                early_stopping_rounds=30,
                direction='maximize',
                stop_callback=partial(log_early_stopping, logger=logger),
            )
        ],
        catch_exceptions=True,
        logger=logger,
        random_state=random_state,
    )


@contextmanager
def get_context_for_repeat(method_name: str, group_id: str, run_index: int, random_state: int):
    with start_run(run_name=method_name) as mlflow:
        with capture_output() as buffer:
            set_tag("method_name", method_name)
            set_tag("random_state", random_state)
            set_tag("run_index", run_index)
            set_tag("group_id", group_id)
            set_tag("nested", True)
            yield

        log_output(buffer())


def report_result(
    X: DataFrame, y: Target, method: Type[Method], method_name: str, result: Result, random_state: int
) -> None:
    log_result(X, y, method, random_state, result)

    metrics_splits = compute_metrics_on_splits(
        result,
        [c_index, make_brier(X, 3 * 365)],
        y,
    )

    for split_name, split_metrics in metrics_splits.items():
        with update_nested_run(run_name=split_name):
            for metric_name, metric_value in split_metrics.items():
                log_metric(metric_name, metric_value)


if __name__ == '__main__':
    main()
