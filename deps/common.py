from numpy import mean, std
from pandas import DataFrame
from typing import Sequence, Tuple, Type, Dict, Hashable

from deps.pipelines import PipelineConfiguration
from hcve_lib.custom_types import Target, Method, Result

from hcve_lib.data import get_survival_y

from deps.data import load_metadata, get_homage_X, load_data
from deps.logger import logger
from deps.memory import memory
from hcve_lib.evaluation_functions import compute_metrics_ci, compute_metrics_result
from hcve_lib.metrics import CIndex
from hcve_lib.tracking import log_pickled, log_model, log_metrics_ci


def get_data(remove_cohorts: Sequence[str] = tuple(), lifetime: bool = False) -> Tuple:
    metadata = load_metadata()
    data = load_data(metadata)

    if len(remove_cohorts) > 0:
        logger.info(f'Manually dropping cohorts: {", ".join(remove_cohorts)}')
        data_selected = data[~data['STUDY'].isin(remove_cohorts)]
        data_selected['STUDY'] = data_selected['STUDY'].cat.remove_unused_categories()
    else:
        data_selected = data

    X, y = (
        get_homage_X(data_selected, metadata),
        get_survival_y(data_selected, 'NFHF', metadata),
    )

    if lifetime:
        y['data']['tte'] += X['AGE']

    return data_selected, metadata, X, y


get_data_cached = memory.cache(get_data)


def log_result(
    X: DataFrame,
    y: Target,
    method: Type[Method],
    configuration: PipelineConfiguration,
    random_state: int,
    result: Result,
    prefix: str = '',
) -> None:
    log_pickled(str(method.get_estimator(X, random_state, configuration)), 'pipeline.txt')
    log_model(result)

    metrics_ci = compute_metrics_ci(
        result,
        [
            CIndex(),  # Brier(X),
        ],
        y,
    )
    log_metrics_ci(metrics_ci, drop_ci=True, prefix=prefix)


def compute_standard_metrics(
    X: DataFrame,
    y: Target,
    result: Result,
) -> Dict[Hashable, Dict[Hashable, float]]:
    return compute_metrics_result(get_standard_metrics(X, y), y, result)


def get_standard_metrics(X, y):
    return [CIndex()]
