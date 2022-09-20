from functools import partial
from typing import Dict, TypeVar, Hashable, Union

import plotly.graph_objects as go
from pandas import DataFrame
from plotly.graph_objs import Figure
from toolz import itemmap, valfilter, valmap

from deps.constants import RANDOM_STATE
from hcve_lib.custom_types import ExceptionValue, Target, Result
from hcve_lib.evaluation_functions import get_splits_by_age, compute_metrics_result, compute_metrics_prediction, \
    is_not_exception_value
from hcve_lib.evaluation_functions import merge_predictions
from hcve_lib.functional import mapl, flatten
from hcve_lib.functional import pipe
from hcve_lib.metrics import BootstrappedMetric
from hcve_lib.metrics import CIndex, StratifiedMetric
from hcve_lib.tracking import load_run_results
from hcve_lib.visualisation import b

HashableT = TypeVar('HashableT', bound=Hashable)


def stratified_c_index(result):
    return pipe(
        result,
        compute_stratified_c_index,
        display_stratified_c_index,
    )


def compute_stratified_c_index(
    result: Result,
    X: DataFrame,
    y: Target,
) -> Dict[HashableT, Dict[str, Union[float, ExceptionValue]]]:
    return compute_metrics_result(
        [
            StratifiedMetric(
                BootstrappedMetric(
                    CIndex(),
                    RANDOM_STATE,
                    iterations=1000,
                ),
                get_splits_by_age(X['AGE']),
            )
        ],
        y,
        result,
    )


def display_stratified_c_index(metrics):
    for test_cohort, age_groups_c_index in metrics.items():
        b(f'{test_cohort=}')
        age_groups_c_index_sorted = sorted(age_groups_c_index.items(), key=lambda i: i[0])
        for age_group, c_index in age_groups_c_index_sorted:
            if not isinstance(c_index, ExceptionValue):
                print(age_group)
                print(c_index)


def get_stratify_analysis_merged(run_id, X, y, iterations=10):
    return compute_metrics_prediction(
        [
            StratifiedMetric(
                BootstrappedMetric(
                    CIndex(),
                    RANDOM_STATE,
                    iterations=iterations,
                    return_summary=False,
                ),
                get_splits_by_age(X['AGE']),
            )
        ],
        y,
        prediction=merge_predictions(load_run_results(run_id)),
    )


def get_stratify_analysis_per_cohort(run_id, X, y, iterations=10):
    return pipe(
        compute_metrics_result(
            [
                StratifiedMetric(
                    BootstrappedMetric(
                        CIndex(),
                        RANDOM_STATE,
                        iterations=iterations,
                        return_summary=False,
                    ),
                    get_splits_by_age(X['AGE']),
                )
            ],
            y,
            load_run_results(run_id),
        ),
        partial(valmap, partial(valfilter, lambda v: not isinstance(v, ExceptionValue))),
    )


def plot_stratify_analysis(
    fig: Figure,
    df: Dict,
    name: str,
    X: DataFrame,
    points: bool = False,
):
    df_no_missing = valfilter(is_not_exception_value, df)
    preprocessed = DataFrame(
        list(
            flatten(
                itemmap(
                    lambda args: (args[0], mapl(lambda value: (args[0], value), args[1])),
                    df_no_missing,
                ).values()
            )
        )
    )

    fig.add_trace(
        go.Violin(
            x=preprocessed[0],
            y=preprocessed[1],
            legendgroup=name,
            scalegroup=name,
            name=name,
            meanline_visible=True,
            points='all' if points else None,
        )
    )

    fig.update_layout(
        violinmode='group',
        yaxis=dict(range=[0.5, 1]),  # labels={"age__30_40__c_index": 'xx'},
    )
