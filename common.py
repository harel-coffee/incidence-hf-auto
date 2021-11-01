from numbers import Rational

import numpy
from sksurv.metrics import brier_score

from hcve_lib.custom_types import FoldPrediction
from hcve_lib.data import to_survival_y_records


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
