import argparse
import logging
from typing import Dict, Any, Callable, Hashable

import mlflow
from pandas import DataFrame

# noinspection PyUnresolvedReferences
from deps.ignore_warnings import *
from deps.logger import logger
from hcve_lib.custom_types import FoldPrediction, Target, SplitInput
from hcve_lib.cv import cross_validate, filter_missing_features


def run_prediction(
    cv: Dict[Any, SplitInput],
    X: DataFrame,
    y: Target,
    get_pipeline: Callable,
    predict: Callable,
    n_jobs=-1,
) -> Dict[Hashable, FoldPrediction]:

    return cross_validate(
        X,
        y,
        get_pipeline,
        predict,
        cv,
        lambda x_train, x_test: filter_missing_features(x_train, x_test),
        n_jobs=n_jobs,
        logger=logger,
    )


def start_method_run(name: str) -> mlflow.ActiveRun:
    run = mlflow.start_run(run_name=name, nested=True)
    mlflow.log_param('run_name', name)
    return run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv', choices=('lco', '10-fold', 'lm', 'reproduce'))
    args = parser.parse_args()
    run_prediction(**vars(args))
