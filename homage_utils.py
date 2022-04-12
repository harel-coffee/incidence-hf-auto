from typing import Dict, Hashable

from pandas import DataFrame

from hcve_lib.custom_types import Target, SplitPrediction, Result
from hcve_lib.evaluation_functions import compute_metrics_on_splits, c_index, brier
from hcve_lib.utils import partial2_args


def compute_standard_metrics(
    X: DataFrame,
    y: Target,
    result: Result,
) -> Dict[Hashable, Dict[Hashable, float]]:
    return compute_metrics_on_splits(
        result,
        get_standard_metrics(X, y),
        y,
    )


def get_standard_metrics(X, y):
    return [
        partial2_args(c_index, kwargs={
            'X': X, 'y': y
        }),
        # TODO: Disabled
        # partial2_args(
        #     brier,
        #     kwargs={
        #         'X': X,
        #         'y': y,
        #         'time_point': 3 * 365,
        #     },
        #     name='brier_3_years',
        # ),
    ]
