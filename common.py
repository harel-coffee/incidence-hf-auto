from typing import Type

from pandas import DataFrame

from hcve_lib.custom_types import Target, Result, Method
from hcve_lib.evaluation_functions import compute_metrics_ci, c_index, make_brier
from hcve_lib.tracking import log_pickled, log_metrics_ci


def log_result(
    X: DataFrame, y: Target, current_method: Type[Method], random_state: int, result: Result, prefix: str = ''
) -> None:
    log_pickled(str(current_method.get_estimator(X, random_state)), 'pipeline.txt')
    log_pickled(result, 'result')

    metrics_ci = compute_metrics_ci(
        result,
        [
            c_index,
            make_brier(
                X,
                time=3 * 365,
            ),
        ],
        y,
    )
    log_metrics_ci(metrics_ci, drop_ci=True, prefix=prefix)
