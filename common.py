from numbers import Rational

from sksurv.metrics import brier_score

from hcve_lib.custom_types import FoldPrediction


def brier(fold: FoldPrediction, time_point: Rational) -> float:
    return brier_score(
            fold['y_train'],
            fold['y_true'],
            [fn(time_point) for fn in fold['model'] \
                .predict_survival_function(fold['X_test'])],
            365*3,
        )[1][0]
