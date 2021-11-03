from numbers import Rational

import numpy
from sksurv.metrics import brier_score

from hcve_lib.custom_types import FoldPrediction
from hcve_lib.data import to_survival_y_records
from hcve_lib.evaluation_functions import compute_metrics_ci, c_index
from hcve_lib.tracking import log_pickled, log_metrics_ci
from hcve_lib.utils import partial2


def brier(fold: FoldPrediction, time_point: Rational) -> float:
    # TODO: Hack
    if not isinstance(fold['y_true'], numpy.recarray):
        y_true = to_survival_y_records(fold['y_true'])
    else:
        y_true = fold['y_true']

    return brier_score(
            fold['y_train'],
            y_true,
            [fn(time_point) for fn in fold['model'] \
                .predict_survival_function(fold['X_test'])],
            365*3,
        )[1][0]


def log_result(X, y, current_method, result):
    brier_3_years = partial2(brier, kwargs={'time_point': 3 * 365})
    metrics_ci = compute_metrics_ci(result['predictions'], [c_index, brier_3_years], y)
    log_pickled(str(current_method['get_estimator'](X)), 'pipeline')
    log_pickled(result, 'result')
    log_metrics_ci(metrics_ci)
