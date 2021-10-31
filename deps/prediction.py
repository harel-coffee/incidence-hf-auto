import argparse
import logging
from typing import Dict, Any, Callable

import mlflow
from pandas import DataFrame

from deps.logger import logger
from hcve_lib.custom_types import FoldPrediction
from hcve_lib.cv import cross_validate, predict_survival, filter_missing_features, FoldInput, Target
# noinspection PyUnresolvedReferences
from deps.ignore_warnings import *


def run_prediction(
    cv: Dict[Any, FoldInput],
    X: DataFrame,
    y: Target,
    get_pipeline: Callable,
) -> Dict[Any, FoldPrediction]:
    logging.basicConfig(
        format='[%(asctime)s] %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO,
    )
    logger.setLevel('INFO')

    for column in X.columns:
        if len(X[column].unique()) < 10:
            X.loc[:, column] = X[column].astype('category')

    return cross_validate(
        X,
        y,
        get_pipeline,
        predict_survival,
        cv,
        lambda x_train, x_test: filter_missing_features(x_train, x_test),
        n_jobs=-1,
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