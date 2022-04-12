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
    formatted_matrix = map_deep(
        matrix,
        mapper=format_matrix_value,
        levels=2,
    )

    numerical_matrix = map_deep(
        matrix,
        lambda value,
        level: (value['mean'] if value['mean'] else 0.5) if value and level == 1 else value,
        levels=2,
    )
    matrix_numerical_df = DataFrame(numerical_matrix)
    matrix_formatted_df = DataFrame(formatted_matrix)

    pass_kwargs = dict(
        **kwargs,
        zmin=matrix_numerical_df.min().min(),
        zmax=matrix_numerical_df.max().max(),
    )

    figure = ff.create_annotated_heatmap(
        matrix_numerical_df.round(2).to_numpy(),
        annotation_text=matrix_formatted_df.to_numpy(),
        x=list(matrix_formatted_df.columns),
        y=list(matrix_formatted_df.index),
        font_colors=['white', 'black'],
        **dict(
            **kwargs,
            zmin=matrix_numerical_df.min().min(),
            zmax=matrix_numerical_df.max().max(),
        ),
    )
    figure.update_layout(xaxis_title="Trained on", yaxis_title='Tested on')
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(showgrid=False)
    return figure


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
