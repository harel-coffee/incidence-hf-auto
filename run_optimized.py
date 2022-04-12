import argparse
from logging import INFO
from typing import List

import mlflow
import optuna
from mlflow import start_run, set_tracking_uri, set_tag

from common import log_result
from deps.common import get_data_cached
# noinspection PyUnresolvedReferences
from hcve_lib.custom_types import Method
from deps.logger import logger
from hcve_lib.cv import Optimize, optimize_per_split, evaluate_optimize_splits, optimize_per_group
from hcve_lib.splitting import get_splitter
from hcve_lib.optimization import EarlyStoppingCallback, optuna_report_mlflow
from hcve_lib.tracking import log_optimizer, update_nested_run
from hcve_lib.utils import partial2
from deps.pipelines import get_pipelines

study = optuna.create_study(direction='maximize')


def run(
    methods: List[str], splits_name: str, n_jobs: int, n_trials: int, remove_cohorts: List[str], group_by: str
) -> None:
    data, metadata, X, y = get_data_cached(remove_cohorts)
    logger.setLevel(INFO)
    logger.info('Data loaded')
    set_tracking_uri('http://localhost:5000')
    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    get_splits = get_splitter(splits_name)

    mlflow.set_experiment('optimized_' + splits_name + (f'_per_{group_by.lower()}' if group_by else ''))
    for method_name in methods:
        method_definition = get_pipelines()[method_name]

        with start_run(run_name=method_name):
            set_tag('removed_cohorts', remove_cohorts)

            if group_by:
                optimizers = optimize_per_group(
                    partial2(
                        get_optimize,
                        get_splits=get_splits,
                        method=method_definition,
                        n_jobs=n_jobs,
                        n_trials=n_trials,
                    ),
                    X=X.groupby(data[group_by]),
                    y=y,
                    mlflow_track=True,
                    n_jobs=n_jobs,
                )

            else:
                optimizers = optimize_per_split(
                    partial2(
                        get_optimize,
                        method=method_definition,
                        n_jobs=n_jobs,
                        n_trials=n_trials,
                    ),
                    get_splits=partial2(get_splits, data=data),
                    X=X,
                    y=y,
                    mlflow_track=True,
                    n_jobs=n_jobs,
                )

                result_splits = {}
                for split_name, optimizer in optimizers.items():
                    result_splits[split_name] = optimizer.study.best_trial.user_attrs['result_split']['train_test']
                log_result(X, y, method_definition, method_name, result_splits)

            for split_name, optimizer in optimizers.items():
                with update_nested_run(run_name=split_name):
                    set_tag("method_name", method_name)
                    log_optimizer(optimizer)


def get_optimize(method: Method, n_jobs: int, n_trials: int, get_splits=None, mlflow_track: bool = False):
    return Optimize(
        partial2(method.get_estimator, verbose=0),
        method.predict,
        method.optuna,
        objective_evaluate=partial2(
            evaluate_optimize_splits,
            log_mlflow=True,
        ),
        mlflow_callback=mlflow_track,
        get_splits=get_splits,
        optimize_params={
            'n_jobs': n_jobs,
            'n_trials': n_trials,
        },
        optimize_callbacks=[
            optuna_report_mlflow, EarlyStoppingCallback(early_stopping_rounds=50, direction='maximize')
        ],
        logger=logger,
        catch_exceptions=True,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('methods', metavar='METHOD', type=str, nargs='+')
    parser.add_argument('--predictions', type=str, dest='splits_name')
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--group-by', type=str, nargs='?')
    parser.add_argument('--remove-cohorts', type=str, default=tuple(), nargs="*")

    args = parser.parse_args()
    run(**vars(args))
