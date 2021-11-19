import argparse
from functools import partial
from logging import INFO
from typing import List

import mlflow
import optuna
from mlflow import start_run, set_tracking_uri, set_tag

from deps.common import get_variables_cached
# noinspection PyUnresolvedReferences
from deps.ignore_warnings import *
from deps.logger import logger
from hcve_lib.cv import Optimize, optimize_simple_cv, get_reproduce_split, evaluate_optimize_splits, \
    train_test_proportion
from hcve_lib.cv import get_lco_splits
from hcve_lib.optimization import EarlyStoppingCallback, optuna_report_mlflow
from hcve_lib.tracking import log_optimizer, update_nested_run
from pipelines import get_pipelines

study = optuna.create_study(direction='maximize')


def run_lco_optimized(methods: List[str], splits: str) -> None:
    data, metadata, X, y = get_variables_cached()
    logger.setLevel(INFO)
    logger.info('Data loaded')
    set_tracking_uri('http://localhost:5000')
    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    if splits == "lco":
        get_cv = get_lco_splits
    elif splits == 'reproduce':
        get_cv = get_reproduce_split
    else:
        raise Exception('Splitting not know')

    mlflow.set_experiment('optimized_' + splits)
    for method_name in methods:
        method_definition = get_pipelines()[method_name]
        with start_run(run_name=method_name):
            optimizers = optimize_simple_cv(
                lambda: Optimize(
                    partial(method_definition.get_estimator, verbose=0),
                    method_definition.predict,
                    method_definition.optuna,
                    objective_evaluate=partial(
                        evaluate_optimize_splits,
                        log_mlflow=True,
                    ),
                    optimize_params={
                        'n_jobs': 10,
                        'n_trials': 100,
                    },
                    optimize_callbacks=[
                        optuna_report_mlflow,
                        EarlyStoppingCallback(early_stopping_rounds=50, direction='maximize')
                    ],
                    logger=logger,
                    catch_exceptions=True,
                ),
                get_cv=partial(get_cv, data=data),
                X=X,
                y=y,
                mlflow_track=True,
                n_jobs=-1,
            )

            for split_name, optimizer in optimizers.items():
                with update_nested_run(run_name=split_name, ):
                    set_tag("method_name", method_name)
                    log_optimizer(optimizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('methods', metavar='METHOD', type=str, nargs='+')
    parser.add_argument('--splits', type=str)
    args = parser.parse_args()
    run_lco_optimized(**vars(args))
