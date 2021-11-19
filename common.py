from numbers import Rational

import numpy
from pandas import DataFrame
from sksurv.metrics import brier_score

from hcve_lib.custom_types import FoldPrediction, Target, SplitInput
from hcve_lib.data import to_survival_y_records
from hcve_lib.evaluation_functions import compute_metrics_ci, c_index
from hcve_lib.tracking import log_pickled, log_metrics_ci
from hcve_lib.utils import partial2, split_data


def brier(fold: FoldPrediction, X: DataFrame, y: Target, time_point: Rational) -> float:
    X_train, y_train, X_test, y_test = split_data(X, y, fold)

    if not isinstance(y_test, numpy.recarray):
        y_test = {**y_test, 'data': to_survival_y_records(y_test)}
    if not isinstance(y_train, numpy.recarray):
        y_train = {**y_train, 'data': to_survival_y_records(y_train)}

    return brier_score(
            y_train['data'],
            y_test['data'],
            [fn(time_point) for fn in fold['model'] \
                .predict_survival_function(X_test)],
            time_point,
        )[1][0]


def brier_y_score(fold: FoldPrediction, X: DataFrame, y: Target, time_point: Rational) -> float:
    X_train, y_train, X_test, y_test = split_data(X, y, fold)
    if not isinstance(y_test, numpy.recarray):
        y_test = {**y_test, 'data': to_survival_y_records(y_test)}
    if not isinstance(y_train, numpy.recarray):
        y_train = {**y_train, 'data': to_survival_y_records(y_train)}

    return brier_score(
        y_train['data'],
        y_test['data'],
        fold['y_score'],
        time_point,
    )[1][0]


def log_result(X, y, current_method, result):
    brier_3_years = partial2(brier, kwargs={'time_point': 3 * 365, 'X': X, 'y': y})
    c_index_ = partial2(c_index, kwargs={'X': X, 'y': y})
    metrics_ci = compute_metrics_ci(
        result,
        [
            c_index_,
            # brier_3_years
        ],
    )
    log_pickled(str(current_method.get_estimator(X)), 'pipeline.txt')
    log_pickled(result, 'result')
    log_metrics_ci(metrics_ci)
