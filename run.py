from deps.data import get_30_to_80
from hcve_lib.data import get_age_range
from hcve_lib.log_output import capture_output, log_output
import argparse
from contextlib import contextmanager
from dataclasses import asdict
from enum import auto
from functools import partial
from logging import INFO
from typing import List, Type, Optional
from uuid import uuid4

import optuna
from mlflow import start_run, set_tracking_uri, set_tag, set_experiment
from pandas import DataFrame

from deps.common import get_data_cached, get_data, log_result
# noinspection PyUnresolvedReferences
from deps.constants import RANDOM_STATE, TRACKING_URI
from deps.evaluation import parse_configuration, get_nested_optimization
from deps.logger import logger as global_logger
from deps.pipelines import get_pipelines, PipelineConfiguration, get_imputer, ImputeOption
from hcve_lib.custom_types import Target, Method, Result, StrEnum
from hcve_lib.cv import repeated_cross_validate
from hcve_lib.evaluation_functions import compute_metrics_result
from hcve_lib.metrics import CIndex
from hcve_lib.splitting import filter_missing_features, get_splitter
from hcve_lib.tracking import update_nested_run, log_metric
from hcve_lib.utils import random_seed, SaveEnum, loc
from hcve_lib.wrapped_sklearn import DFPipeline


class Analysis(StrEnum):
    DEFAULT = auto()
    NESTED_OPTIMIZATION = auto()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('methods', metavar='METHOD', type=str, nargs='+')
    parser.add_argument('--configuration', type=str, default=None)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--n-repeats', type=int, default=1)
    parser.add_argument('--n-trials', type=int, default=30)
    parser.add_argument('--splits', type=str, dest='splits_name')
    parser.add_argument('--analysis', type=Analysis, action=SaveEnum, default=Analysis.DEFAULT)
    parser.add_argument('--objective-metric', type=str, default='c_index')
    args = parser.parse_args()
    run(**vars(args))


def run(
    methods: List[str],
    configuration: Optional[str],
    n_repeats: int,
    n_jobs: int,
    n_trials: int,
    splits_name: str,
    analysis: Analysis,
    objective_metric: str,
) -> None:
    configuration_ = parse_configuration(configuration)

    group_id = uuid4()
    random_seed(RANDOM_STATE)
    set_tracking_uri(TRACKING_URI)

    global_logger.setLevel(INFO)
    global_logger.info('Data loaded')
    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    get_splits = get_splitter(splits_name)

    data, metadata, X, y = get_data()
    if configuration_.age_range:
        X = get_age_range(X, configuration_.age_range)
        data = loc(X.index, data)
        y = loc(X.index, y)

    for method_name in methods:
        method = get_pipelines()[method_name]

        if method_name == 'pcp_hf':
            X = DFPipeline(get_imputer(X, ImputeOption.SIMPLE)).fit_transform(X, y)

        if analysis is Analysis.NESTED_OPTIMIZATION:
            set_experiment(f'optimized_nested_{splits_name}')
            get_pipeline = partial(
                get_nested_optimization,
                configuration=configuration_,
                data=data,
                method=method,
                n_trials=n_trials,
                y=y,
                log=True,
                logger=global_logger,
                objective_metric=objective_metric,
            )
        elif analysis is Analysis.DEFAULT:
            set_experiment(f'default_{splits_name}')
            get_pipeline = partial(method.get_estimator, configuration=configuration_)
        else:
            raise Exception(f'Analysis \'{analysis.value}\' not supported')

        results: List[Result] = repeated_cross_validate(
            get_pipeline=get_pipeline,
            method=method,
            n_repeats=n_repeats,
            train_test_filter_callback=filter_missing_features,
            X=X,
            y=y,
            get_splits=partial(get_splits, data=data),
            get_repeat_mlflow_context=partial(get_context_for_repeat, method_name, configuration_, group_id),
            on_repeat_result=partial(report_result, X, y, method, configuration_, method_name),
            n_jobs=n_jobs,
            mlflow_track=True,
            random_state=RANDOM_STATE,
        )


@contextmanager
def get_context_for_repeat(
    method_name: str,
    configuration: PipelineConfiguration,
    group_id: str,
    run_index: int,
    random_state: int,
):
    with start_run(run_name=method_name) as mlflow:
        with capture_output() as buffer:
            for key, value in asdict(configuration).items():
                set_tag(key, value)
            set_tag("method_name", method_name)
            set_tag("random_state", random_state)
            set_tag("run_index", run_index)
            set_tag("group_id", group_id)
            set_tag("nested", True)
            yield

        log_output(buffer())


def report_result(
    X: DataFrame,
    y: Target,
    method: Type[Method],
    configuration: PipelineConfiguration,
    method_name: str,
    result: Result,
    random_state: int,
) -> None:
    log_result(X, y, method, configuration, random_state, result)

    metrics_splits = compute_metrics_result(
        [
            CIndex(),  # Brier(X),
            # StratifiedMetric(SimpleBrier(X), get_splits_by_class(y)),
        ],
        y,
        result
    )

    for split_name, split_metrics in metrics_splits.items():
        with update_nested_run(run_name=split_name):
            for metric_name, metric_value in split_metrics.items():
                log_metric(metric_name, metric_value)


if __name__ == '__main__':
    main()
