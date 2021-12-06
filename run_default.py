import argparse
import logging
from functools import partial
from typing import List, Hashable, Dict, Callable

from mlflow import start_run, set_tracking_uri, set_tag, set_experiment
from pandas import DataFrame

from common import log_result
from deps.common import get_variables_cached
# noinspection PyUnresolvedReferences
from deps.custom_types import Method
from deps.logger import logger
from deps.tracking import log_splits_as_subruns
from hcve_lib.custom_types import Target, SplitPrediction
from hcve_lib.cv import cross_validate, execute_per_group
from hcve_lib.evaluation_functions import compute_metrics_ci
from hcve_lib.splitting import get_splitter
from deps.pipelines import get_pipelines
from utils import compute_standard_metrics, get_standard_metrics


def run(
    selected_methods: List[str], splitter_name: str, remove_cohorts: List[str], group_by: str,
    n_jobs: int
) -> None:
    set_tracking_uri('http://localhost:5000')
    logger.setLevel(logging.INFO)
    set_experiment(splitter_name + '_default')
    data, metadata, X, y = get_variables_cached(remove_cohorts)
    get_splits = get_splitter(splitter_name)
    for method_name in selected_methods:
        with start_run(run_name=method_name):
            set_tag('removed_cohorts', remove_cohorts)
            method = get_pipelines()[method_name]
            if group_by:
                results_per_group = execute_per_group(
                    partial(execute_group, get_splits=get_splits, method=method),
                    X.groupby(data[group_by]),
                    y,
                    n_jobs,
                )
                metrics_per_group = {
                    name: compute_metrics_ci(result, get_standard_metrics(X, y))
                    for name, result in results_per_group.items()
                }
                log_splits_as_subruns(metrics_per_group)
            else:
                result = cross_validate(
                    X,
                    y,
                    method.get_estimator,
                    method.predict,
                    get_splits(X=X, y=y, data=data),
                    n_jobs=-1,
                    logger=logger,
                )
                log_result(X, y, method, method_name, result)
                metrics_splits = compute_standard_metrics(X, y, result)
                log_splits_as_subruns(metrics_splits)


def execute_group(_: str, X: DataFrame, y: Target, get_splits: Callable,
                  method: Method) -> Dict[Hashable, SplitPrediction]:
    return cross_validate(
        X,
        y,
        method.get_estimator,
        method.predict,
        get_splits(X=X, y=y),
        n_jobs=1,
        logger=logger,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('selected_methods', metavar='METHOD', type=str, nargs='+')
    parser.add_argument('--splits', dest='splitter_name')
    parser.add_argument('--remove-cohorts', type=str, default=tuple(), nargs="*")
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--group-by', type=str, nargs='?')
    args = parser.parse_args()
    run(**vars(args))
