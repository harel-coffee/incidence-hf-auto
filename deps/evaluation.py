from ast import literal_eval as make_tuple
from collections import defaultdict
from functools import partial
from typing import Any, List, Dict, Optional
from typing import Tuple

import mlflow
from mlflow import get_experiment_by_name
from mlflow.entities import Run
from pandas import DataFrame
from toolz import keymap, valmap, compose

from hcve_lib.custom_types import ValueWithCI
from hcve_lib.evaluation_functions import compute_metric_ci
from hcve_lib.functional import pipe
from hcve_lib.tracking import display_run_info_context
from hcve_lib.utils import transpose_dict
from hcve_lib.visualisation import b


def get_metric_matrix(metric_name: str = 'c_index', method_name: str = 'gb', show_info: bool = False) -> DataFrame:
    chosen_run_lm, latest_lm = get_latest_chosen_run_group('optimized_nested_lm', method_name)
    if show_info and chosen_run_lm:
        b('LM')
        display_run_info_context(chosen_run_lm.info.run_id)

    lm_metric_value_by_train_test = pipe(
        latest_lm,
        partial(get_metrics_from_group, metric_name=metric_name),
        partial(keymap, make_tuple),
        key_pairs_to_nested,
    )

    parent_run_lco, group_of_runs = get_latest_chosen_run_group('optimized_nested_lco', method_name)

    if parent_run_lco and group_of_runs:
        if show_info:
            b('LCO')
            display_run_info_context(parent_run_lco.info.run_id)

        lco_metrics = get_metrics_from_group(group_of_runs, metric_name)
    else:
        lco_metrics = None

    return {**lm_metric_value_by_train_test, 'Centralized': lco_metrics}
    # matrix = pandas.concat(
    #     [
    #         DataFrame(lm_metric_value_by_train_test),
    #         DataFrame({'LCO': lco_metrics}),
    #     ],
    #     axis=1,
    # )
    # matrix = matrix.iloc[::-1]


def get_metrics_from_group(
    group_of_runs: List[List[Run]],
    metric_name: str,
) -> Optional[Dict[str, ValueWithCI]]:
    return pipe(
        {nr: get_metric_from_runs(metric_name, runs)
         for nr, runs in enumerate(group_of_runs)},
        transpose_dict,
        partial(valmap, compose(compute_metric_ci, dict.values)),
    )


def get_metric_from_runs(
    metric_name: str,
    runs: List[Run],
) -> Dict[str, float]:
    return {run.data.tags['mlflow.runName']: run.data.metrics.get(metric_name) for run in runs}


def get_latest_chosen(
    experiment_name: str,
    method_name: str,
) -> Tuple[Optional[Run], Optional[List[Run]]]:
    optimized_lco_id = get_experiment_by_name(experiment_name).experiment_id
    parent_runs = mlflow.search_runs(
        optimized_lco_id,
        f'tags.chosen = "True" and tags.mlflow.runName = "{method_name}"',
        output_format='list',
        max_results=1,
    )

    if len(parent_runs) == 0:
        return None, None
    else:
        parent_run = parent_runs[0]

    search_filter = f'tags.mlflow.parentRunId = "{parent_run.info.run_id}"'

    return parent_run, mlflow.search_runs(
        get_experiment_by_name(experiment_name).experiment_id, search_filter, output_format='list'
    )


def get_latest_chosen_run_group(
    experiment_name: str,
    method_name: str,
) -> Tuple[Optional[Run], Optional[List[List[Run]]]]:
    optimized_lco_id = get_experiment_by_name(experiment_name).experiment_id
    root_runs = mlflow.search_runs(
        optimized_lco_id,
        "tags.chosen = 'True' and tags.mlflow.runName = '" + method_name + "'",
        output_format='list',
        max_results=1,
    )
    if len(root_runs) == 0:
        return None, None
    else:
        root_run = root_runs[0]

    return root_run, get_runs_and_subruns_in_group(root_run.data.tags['group_id'], experiment_name)


def get_runs_and_subruns_in_group(
    group_id: str,
    experiment_name: str,
) -> List[List[Run]]:
    runs_in_group = get_runs_in_group(group_id, experiment_name)

    output_list = []

    for run in runs_in_group:
        output_list.append(
            mlflow.search_runs(
                get_experiment_by_name(experiment_name).experiment_id,
                f'tags.mlflow.parentRunId = "{run.info.run_id}"',
                output_format='list'
            )
        )

    return output_list


def get_runs_in_group(
    group_id: str,
    experiment_name: str,
) -> List[Run]:
    return mlflow.search_runs(
        get_experiment_by_name(experiment_name).experiment_id,
        f'tags.group_id = "{group_id}"',
        output_format='list',
    )


def key_pairs_to_nested(input: Dict[Tuple, Any]):
    result: Dict = defaultdict(dict)

    keys_train = set(k[0] for k in input.keys())
    keys_test = set(k[1] for k in input.keys())

    for cohort_train in keys_train:
        for cohort_test in keys_test:
            result[cohort_train][cohort_test] = input.get((cohort_train, cohort_test))

    return result
