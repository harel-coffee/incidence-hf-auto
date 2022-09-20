from typing import Dict, Any

import plotly.figure_factory as ff
from matplotlib.figure import Figure
from pandas import DataFrame

from hcve_lib.custom_types import ValueWithCI
from hcve_lib.functional import map_recursive, t, map_deep


def plot_metric_matrix(
    matrix: Dict[str, Dict[str, ValueWithCI]],
    **kwargs,
) -> Figure:

    matrix_numerical_df = get_numerical_matrix_df(matrix)

    formatted_matrix = map_deep(
        matrix,
        mapper=format_matrix_value,
        levels=2,
    )

    matrix_formatted_df = DataFrame(formatted_matrix)

    figure = plot_matrix(matrix_numerical_df, matrix_formatted_df, **kwargs)

    return figure


def plot_matrix(
    matrix: DataFrame,
    labels: DataFrame = None,
    **kwargs,
) -> Figure:

    pass_kwargs = dict(
        **kwargs,
        zmin=matrix.min().min(),
        zmax=matrix.max().max(),
    )
    figure = ff.create_annotated_heatmap(
        matrix.round(2).to_numpy(),
        annotation_text=None if labels is None else labels.to_numpy(),
        x=list(matrix.columns),
        y=list(matrix.index),
        font_colors=['white', 'black'],
        **dict(
            **kwargs,
            zmin=matrix.min().min(),
            zmax=matrix.max().max(),
        ),
    )
    figure.update_layout(xaxis_title="Trained on", yaxis_title='Tested on')
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(showgrid=False)
    return figure


def get_numerical_matrix_df(matrix: Dict[str, Dict[str, ValueWithCI]]) -> DataFrame:
    numerical_matrix = get_numerical_matrix(matrix)
    matrix_numerical_df = DataFrame(numerical_matrix)
    return matrix_numerical_df


def get_numerical_matrix(matrix: Dict[str, Dict[str, ValueWithCI]]) -> Dict[str, Dict[str, ValueWithCI]]:
    numerical_matrix = map_deep(
        matrix,
        (lambda value, level: (value['mean'] if value['mean'] else 0.5) if value and level == 1 else value),
        levels=2,
    )
    return numerical_matrix


def format_value_with_ci(value: ValueWithCI) -> str:
    return f'{value["mean"]:0.2f} {value["ci"][0]:.2f}-{value["ci"][1]:.2f}'


def format_value_with_ci_short(value: ValueWithCI) -> str:
    mean = f'{value["mean"]:0.2f}'.lstrip('0')
    ci = f'{value["ci"][1]-value["ci"][0]:.2f}'.lstrip('0')
    return f'{mean}±{ci} CI'


def format_matrix_value(value: Any, _: int) -> str:
    if value is None:
        return ''
    else:
        return format_value_with_ci_short(value)
