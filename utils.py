from typing import Dict, Hashable

from pandas import DataFrame

from common import brier
from hcve_lib.custom_types import Target, FoldPrediction
from hcve_lib.evaluation_functions import compute_metrics_folds, c_index
from hcve_lib.utils import partial2_args


def compute_standard_metrics(
    X: DataFrame,
    y: Target,
    result: Dict[Hashable, FoldPrediction],
) -> Dict[Hashable, Dict[Hashable, float]]:
    return compute_metrics_folds(
        result,
        [
            partial2_args(c_index, kwargs={
                'X': X,
                'y': y
            }),
            partial2_args(
                brier,
                kwargs={
                    'X': X,
                    'y': y,
                    'time_point': 3 * 365,
                },
                name='brier_3_years',
            ),
        ],
    )
