from hcve_lib.custom_types import Metric, Target
from hcve_lib.evaluation_functions import get_splits_by_class
from hcve_lib.metrics import StratifiedMetric, SimpleBrier, CIndex


def get_metric(
    name: str,
    y: Target,
) -> Metric:
    if name == 'brier_cases':
        return StratifiedMetric(SimpleBrier(), get_splits_by_class(y, [1]))
    elif name == 'c_index':
        return CIndex()
    else:
        raise ValueError(f'Objective metric \'{name}\' not defined')
