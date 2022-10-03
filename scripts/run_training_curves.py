import pickle
from functools import partial
from typing import Dict

from pandas import DataFrame
from plotly import express as px

from deps.common import get_data_cached
from hcve_lib.custom_types import Result, Target
from hcve_lib.evaluation_functions import c_index
from hcve_lib.functional import pipe
from run_training_curves import get_training_curve_file
from typing import Dict
from hcve_lib.custom_types import Result
from run_training_curves import get_training_curve_file
import pickle
from hcve_lib.functional import t, pipe
import line_profiler

profile = line_profiler.LineProfiler()


@profile
def get_training_curves_data(method_name: str):
    with open(get_training_curve_file(method_name), 'rb') as f:
        return dict(pickle.load(f, ))


@profile
def get_training_curves_c_indexes(results: Dict[int, Result], X: DataFrame, y: Target):
    return {
        split: {n_features: c_index(result, X, y)
                for n_features, result in results.items()}
        for split,
        results in results.items()
    }


@profile
def plot_training_curves(c_index_per_n_features: Dict[str, float]):
    fig = px.line(DataFrame(c_index_per_n_features).iloc[::-1])
    # fig.for_each_trace(lambda line: fig.add_annotation(
    #     x=line.x[0], y=line.y[0], text=line.name,
    #     font_color=line.line.color, ax=0, ay=0, xanchor="left", showarrow=False
    # ))
    fig.update_layout(
        showlegend=False,
        xaxis_title="n features",
        yaxis_title='c-statistic',
    )


def run_training_curves(method_name: str, X: DataFrame, y: Target):
    return pipe(
        method_name,
        get_training_curves_data,
        partial(get_training_curves_c_indexes, X=X, y=y),
        plot_training_curves,
    )


data, metadata, X, y = get_data_cached()
run_training_curves('gb', X, y)
