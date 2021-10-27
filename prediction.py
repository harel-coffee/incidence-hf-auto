from typing import Dict, Any, Callable

import argparse
import logging
import mlflow
import pandas
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.preprocessing import FunctionTransformer
from sksurv.linear_model import CoxnetSurvivalAnalysis

from deps.data import load_metadata, load_data_cached
from deps.logger import logger
from hcve_lib.custom_types import FoldPrediction
from hcve_lib.cv import cross_validate, predict_survival, kfold_cv, filter_missing_features, lm_cv, lco_cv, train_test, \
    FoldInput
from hcve_lib.data import to_survival_y_records, get_X, get_survival_y
from hcve_lib.evaluation_functions import compute_metrics_folds, c_index
from hcve_lib.functional import pipe
from hcve_lib.tracking import log_pickled
from hcve_lib.utils import remove_column_prefix
from hcve_lib.wrapped_sklearn import DFColumnTransformer, DFPipeline, DFSimpleImputer, DFOrdinalEncoder, \
    DFCoxnetSurvivalAnalysis
# noinspection PyUnresolvedReferences
from ignore_warnings import *


def run(
    cv: Dict[Any, FoldInput],
    data: DataFrame,
    metadata: Dict,
    get_pipeline: Callable,
) -> Dict[Any, FoldPrediction]:
    logging.basicConfig(
        format='[%(asctime)s] %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO,
    )
    logger.setLevel('INFO')

    X, y = pipe(
        (
            get_X(data, metadata),
            to_survival_y_records(get_survival_y(data, 'NFHF', metadata)),
        ),
    )

    for column in X.columns:
        if len(X[column].unique()) < 10:
            X.loc[:, column] = X[column].astype('category')
        # print(f'{X[column].dtype}: {format_identifier_long(column, metadata)}')

    # if cv == 'reproduce':
    #     cv = train_test(
    #         data,
    #         train_filter=lambda _data: _data['STUDY'].isin(['HEALTHABC', 'PREDICTOR', 'PROSPER']),
    #         test_filter=lambda _data: _data['STUDY'] == 'ASCOT'
    #     )
    # elif args.cv == 'lco':
    #     cv = lco_cv(data.groupby('STUDY'))
    # elif args.cv == '10-fold':
    #     cv = kfold_cv(data, n_splits=10, shuffle=True, random_state=243)
    # elif args.cv == 'lm':
    #     cv = lm_cv(data.groupby('STUDY'))
    # else:
    #     raise Exception('No CV')

    return cross_validate(
        X,
        y,
        get_pipeline,
        predict_survival,
        cv,
        lambda x_train, x_test: filter_missing_features(x_train, x_test),
        # n_jobs=1,
    )


def start_method_run(name: str) -> mlflow.ActiveRun:
    # logger.info(f'\t- {name}')
    run = mlflow.start_run(run_name=name, nested=True)
    mlflow.log_param('run_name', name)
    return run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv', choices=('lco', '10-fold', 'lm', 'reproduce'))
    args = parser.parse_args()
    run(**vars(args))
