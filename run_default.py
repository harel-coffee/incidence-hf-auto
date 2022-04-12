import argparse
import logging
from functools import partial
from typing import List, Hashable, Dict, Callable, Type, Optional

from mlflow import start_run, set_tracking_uri, set_tag, set_experiment
from pandas import DataFrame, Index, Series

from common import log_result
from deps.common import get_data_cached
# noinspection PyUnresolvedReferences
from deps.hyperparameters import get_hyperparameters
from deps.logger import logger
from deps.tracking import log_splits_as_subruns
from hcve_lib.custom_types import Target, SplitPrediction, Splitter, Method
from hcve_lib.cv import cross_validate, execute_per_group
from hcve_lib.evaluation_functions import compute_metrics_ci
from hcve_lib.splitting import get_splitter, get_group_indexes, filter_missing_features
from deps.pipelines import get_pipelines
from homage_utils import compute_standard_metrics, get_standard_metrics


def run(
    selected_methods: List[str],
    splitter_name: str,
    remove_cohorts: List[str],
    group_by: str,
    n_jobs: int,
    pre_optimized: bool,
    federated_groups_name: Optional[str],
    disable_filter_features: bool
) -> None:
    logger.setLevel(logging.INFO)

    set_tracking_uri('http://localhost:5000')
    experiment_prefix = 'default_' if not pre_optimized else 'optimized_'
    set_experiment(experiment_prefix + splitter_name + (f'_per_{group_by.lower()}' if group_by else ''))
    print(experiment_prefix + splitter_name + (f'_per_{group_by.lower()}' if group_by else ''))
    data, metadata, X, y = get_data_cached(remove_cohorts)
    get_splits = get_splitter(splitter_name)

    for method_name in selected_methods:
        logger.info(f'Running {method_name}')

        with start_run(run_name=method_name):
            set_tag('removed_cohorts', remove_cohorts)
            method = get_pipelines()[method_name]

            if federated_groups_name:
                federated_groups = get_group_indexes(data, 'STUDY')
            else:
                federated_groups = None

            if group_by:
                if pre_optimized:
                    raise Exception('pre_optimized not for grouped')

                results_per_group = execute_per_group(
                    partial(
                        run_cross_validate,
                        get_splits=get_splits,
                        method=method,
                        n_jobs=1,
                        federated_groups=federated_groups
                    ),
                    X.groupby(data[group_by]),
                    y,
                    n_jobs,
                )

                metrics_per_group = {
                    name: compute_metrics_ci(result, get_standard_metrics(X, y))
                    for name,
                    result in results_per_group.items()
                }

                log_splits_as_subruns(metrics_per_group)

            else:
                train_test_filter_callback = None if disable_filter_features else filter_missing_features

                result = run_cross_validate(
                    X,
                    y,
                    partial(get_splits, data=data),
                    method,
                    n_jobs=n_jobs,
                    split_hyperparameters=get_hyperparameters()[splitter_name][method_name] if pre_optimized else None,
                    federated_groups=federated_groups,
                    train_test_filter_callback=train_test_filter_callback
                )

                log_result(X, y, method, method_name, result)
                metrics_splits = compute_standard_metrics(X, y, result)
                log_splits_as_subruns(metrics_splits)


def run_cross_validate(
    X: DataFrame,
    y: Target,
    get_splits: Callable,
    method: Type[Method],
    n_jobs: int = -1,
    split_hyperparameters: Dict[str, Dict] = None,
    federated_groups: Dict[str, Index] = None,
    train_test_filter_callback: Callable[[Series, Series], bool] = None,
) -> Result:

    get_estimator: Callable

    if federated_groups:
        get_estimator = partial(method.get_estimator, federated_groups=federated_groups)
    else:
        get_estimator = method.get_estimator

    return cross_validate(
        X,
        y,
        get_estimator,
        method.predict,
        get_splits(X=X, y=y),
        n_jobs=n_jobs,
        logger=logger,
        split_hyperparameters=split_hyperparameters,
        train_test_filter_callback=train_test_filter_callback,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('selected_methods', metavar='METHOD', type=str, nargs='+')
    parser.add_argument('--predictions', dest='splitter_name')
    parser.add_argument('--remove-cohorts', type=str, default=tuple(), nargs="*")
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--group-by', type=str, nargs='?')
    parser.add_argument('--pre-optimized', action='store_true')
    parser.add_argument('--disable-filter-features', action='store_true')
    parser.add_argument('--federated-groups', type=str, dest='federated_groups_name', default=None)
    args = parser.parse_args()
    run(**vars(args))
