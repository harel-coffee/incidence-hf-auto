import json
from ast import literal_eval as make_tuple
from collections import defaultdict
from functools import partial
from logging import Logger
from typing import Any, List, Dict, Optional
from typing import Tuple

import mlflow
from dacite import from_dict, Config
from hcve_lib.custom_types import ValueWithCI, Method, Target
from hcve_lib.cv import OptimizeEstimator, evaluate_optimize_splits
from hcve_lib.evaluation_functions import compute_metric_ci
from hcve_lib.functional import pipe
from hcve_lib.optimization import optuna_report_mlflow, EarlyStoppingCallback
from hcve_lib.splitting import get_train_test
from hcve_lib.tracking import display_run_info_context, log_early_stopping
from hcve_lib.utils import transpose_dict, partial2
from hcve_lib.visualisation import b
from mlflow import get_experiment_by_name, set_tag
from mlflow.entities import Run
from pandas import DataFrame
from toolz import keymap, valmap, compose

from deps.metrics import get_metric
from deps.pipelines import PipelineConfiguration, ResamplingOption, ImputeOption


def get_metric_matrix(
    lm_group: List[List[Run]],
    lco_group: List[List[Run]],
    metric_name: str = 'c_index',
    show_info: bool = False,
) -> Dict:

    lm_metrics = get_metric_matrix_lm(lm_group, metric_name, show_info)

    lco_metrics = get_metric_matrix_lco(lco_group, metric_name, show_info)

    return {**lm_metrics, 'All': lco_metrics}


def get_metric_matrix_lm(
    lm_group: List[List[Run]],
    metric_name: str,
    show_info: bool = False,
):
    chosen_run_lm = lm_group[0][0]
    if show_info and chosen_run_lm:
        b('LM')
        display_run_info_context(chosen_run_lm.info.run_id)
    lm_metric_value_by_train_test = pipe(
        lm_group,
        partial(get_metrics_from_group, metric_name=metric_name),
        partial(keymap, make_tuple),
        key_pairs_to_nested,
    )
    return lm_metric_value_by_train_test


def get_metric_matrix_lco(
    lco_group: List[List[Run]],
    metric_name: str,
    show_info: bool = False,
):
    chosen_run_lco = lco_group[0][0]
    if chosen_run_lco and lco_group:
        if show_info:
            b('LCO')
            display_run_info_context(chosen_run_lco.info.run_id)

        lco_metrics = get_metrics_from_group(lco_group, metric_name)
    else:
        lco_metrics = None
    return lco_metrics


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
    filter_string: str = None,
) -> Optional[List[List[Run]]]:
    experiment_id = get_experiment_by_name(experiment_name).experiment_id
    extra_filter = f' AND {filter_string}' if filter_string is not None else ''

    root_runs = mlflow.search_runs(
        experiment_id,
        "tags.chosen = 'True' and tags.mlflow.runName = '" + method_name + "'" + extra_filter,
        output_format='list',
        max_results=1,
    )

    if len(root_runs) == 0:
        return ValueError('No runs found')
    else:
        root_run = root_runs[0]

    return get_subruns_in_group(root_run.data.tags['group_id'])


def get_subruns_in_group(group_id: str) -> List[List[Run]]:
    runs_in_group = get_runs_in_group(group_id)
    output_list = []
    for run in runs_in_group:
        output_list.append(
            mlflow.search_runs(
                filter_string=f'tags.mlflow.parentRunId = "{run.info.run_id}"',
                output_format='list',
                search_all_experiments=True,
            )
        )

    return output_list


def get_runs_in_group(group_id: str) -> List[Run]:
    return mlflow.search_runs(
        filter_string=f'tags.group_id = "{group_id}"',
        output_format='list',
        search_all_experiments=True,
    )


def key_pairs_to_nested(input: Dict[Tuple, Any]):
    result: Dict = defaultdict(dict)

    keys_train = set(k[0] for k in input.keys())
    keys_test = set(k[1] for k in input.keys())

    for cohort_train in keys_train:
        for cohort_test in keys_test:
            result[cohort_train][cohort_test] = input.get((cohort_train, cohort_test))

    return result


def parse_configuration(configuration: str) -> PipelineConfiguration:
    return from_dict(
        data_class=PipelineConfiguration,
        data=json.loads(configuration) if configuration else {},
        config=Config(cast=[ResamplingOption, ImputeOption]),
    )


def get_nested_optimization(
    X: DataFrame,
    random_state: int,
    configuration: PipelineConfiguration,
    data: DataFrame,
    method: Method,
    n_trials: int,
    y: Target,
    log: bool = False,
    logger: Logger = None,
    objective_metric: str = 'c_index',
):
    set_tag('objective_metric', objective_metric)

    objective_metric_ = get_metric(objective_metric, y)
    direction = objective_metric_.get_direction().value.lower()

    return OptimizeEstimator(
        partial(method.get_estimator, configuration=configuration),
        method,
        method.optuna,
        objective_evaluate=partial(
            evaluate_optimize_splits,
            objective_metric=objective_metric_,
        ),
        optimize_params={
            'n_jobs': 1,
            'n_trials': n_trials,
        },
        get_splits=partial2(get_train_test, randon_state=random_state, data=data),
        optimize_callbacks=[
            *([optuna_report_mlflow] if log else []),
            EarlyStoppingCallback(
                early_stopping_rounds=30,
                direction=direction,
                stop_callback=partial(log_early_stopping, logger=logger),
            )
        ],
        logger=logger,
        random_state=random_state,
        direction=direction
    )
