import logging
from typing import List

import mlflow
from mlflow import start_run, log_metrics
from mlflow.entities import Run

from deps.common import get_data_cached
from deps.evaluation import get_runs_in_group
from hcve_lib.custom_types import ComputeMetric
from hcve_lib.evaluation_functions import compute_metrics_on_splits, compute_metrics_ci, make_brier
from hcve_lib.tracking import load_run_results, get_experiment_id
from hcve_lib.tracking import log_metrics_ci, update_nested_run


def update_run_metrics(run_id: str, metrics: List[ComputeMetric]) -> None:
    try:
        result = load_run_results(run_id)
    except OSError:
        logging.warning(f'Not found for {run_id}')
        return

    missing_metrics = list(set(result['metrics'].keys()) - set(metrics))

    print(f'{missing_metrics=}')

    if len(missing_metrics) == 0:
        return

    # Root run
    metrics_ci = compute_metrics_ci(
        result,
        missing_metrics,
        y,
    )

    log_metrics_ci(metrics_ci, drop_ci=True)

    # Sub runs
    metrics_splits = compute_metrics_on_splits(
        result,
        metrics,
        y,
    )

    for split_name, split_metrics in metrics_splits.items():
        with update_nested_run(run_name=split_name):
            log_metrics(split_metrics)


def search_runs_in_experiment(experiment_name: str, query: str = '', *args, **kwargs) -> List[Run]:
    return mlflow.search_runs(get_experiment_id(experiment_name), query, output_format='list', *args, **kwargs)


if __name__ == '__main__':

    data, metadata, X, y = get_data_cached()

    for experiment in ('optimized_nested_lco', 'optimized_nested_lm'):
        mlflow.set_experiment(experiment)
        for i, root_run in enumerate(search_runs_in_experiment(experiment, "tags.chosen = 'True'")):
            print(f'{i+1}. {root_run.data.tags["mlflow.runName"]} ({root_run.info.run_id}) ', end='')
            if 'group_id' in root_run.data.tags:
                for run in get_runs_in_group(root_run.data.tags['group_id'], experiment):
                    print('.', end='')
                    with start_run(run.info.run_id):
                        try:
                            update_run_metrics(run.info.run_id, [make_brier(X=X, time=3 * 365)])
                        except Exception as e:
                            print(e)
                print()
            else:
                logging.warning('No group_id.')
