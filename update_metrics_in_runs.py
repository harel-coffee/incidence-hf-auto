import argparse
import logging
import traceback

from sklearn.metrics import f1_score, roc_auc_score
from typing import List

import mlflow
from mlflow import start_run
from mlflow.entities import Run
from toolz import valmap

from deps.common import get_data_cached
from deps.constants import RANDOM_STATE
from deps.evaluation import get_runs_in_group
from deps.logger import logger
from hcve_lib.custom_types import Target, Result, Metric
from hcve_lib.evaluation_functions import compute_metrics_result, compute_metrics_ci, get_splits_by_class
from hcve_lib.metrics import StratifiedMetric, SimpleBrier, BinaryMetricAtTime, Brier
from hcve_lib.tracking import load_run_results, get_experiment_id, log_metrics
from hcve_lib.tracking import log_metrics_ci, update_nested_run

from deps.pipelines import DeepCoxMixturePipeline
from hcve_lib.utils import partial2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', type=str, nargs='?')
    parser.add_argument('--force', type=bool, nargs='?', default=False)
    args = parser.parse_args()

    data, metadata, X, y = get_data_cached()
    logger.setLevel(logging.ERROR)

    if args.run_id:
        with start_run(args.run_id):
            update_run_metrics(
                args.run_id,
                [
                    # CIndex(),
                    # Brier(X),
                    # BinaryMetricAtTime(f1_score, time=5 * 365, threshold=0.1)
                    # SimpleBrier(X),
                    StratifiedMetric(SimpleBrier(X), get_splits_by_class(y)),
                    # BinaryMetricAtTime(partial2(roc_auc_score)),
                ],
                y,
                force=args.force
            )
    else:
        for experiment in ('optimized_nested_lco', 'optimized_nested_lm'):
            mlflow.set_experiment(experiment)
            print(experiment.upper())
            for i, root_run in enumerate(search_runs_in_experiment(experiment, "tags.chosen = 'True'")):
                print(f'{i + 1}. {root_run.data.tags["mlflow.runName"]} ({root_run.info.run_id}) ', end='')
                if 'group_id' in root_run.data.tags:
                    for run in get_runs_in_group(root_run.data.tags['group_id']):
                        print('.')
                        with start_run(run.info.run_id):
                            # noinspection PyBroadException
                            try:
                                update_run_metrics(
                                    run.info.run_id,
                                    [
                                        # CIndex(),
                                        # Brier(X),
                                        SimpleBrier(X),  # StratifiedMetric(SimpleBrier(X), get_splits_by_class(y)),
                                    ],
                                    y,
                                    force=args.force,
                                )
                            except Exception as e:
                                traceback.print_exc()
                    print()
                else:
                    logger.warning('No group_id.')


def update_run_metrics(
    run_id: str,
    metrics: List[Metric],
    y: Target,
    force: bool = False,
) -> None:
    try:
        predictions: Result = load_run_results(run_id)
    except OSError:
        logger.warning(f'Not found for {run_id}')
        return

    run: Run = mlflow.get_run(run_id)

    print(run.data.tags['mlflow.runName'])

    if force:
        present_metrics = []
    else:
        present_metrics = list((run.data.metrics.keys()))

    metrics_ci = compute_metrics_ci(
        predictions,
        metrics,
        y,
        skip_metrics=present_metrics,
    )

    log_metrics_ci(metrics_ci, drop_ci=True)

    # Sub runs
    metrics_splits = compute_metrics_result(metrics, y, predictions, skip_metrics=present_metrics)

    for split_name, split_metrics in metrics_splits.items():
        with update_nested_run(run_name=split_name):
            print(f' - {split_name}')
            log_metrics(split_metrics)


def search_runs_in_experiment(experiment_name: str, query: str = '', *args, **kwargs) -> List[Run]:
    return mlflow.search_runs(
        get_experiment_id(experiment_name),
        query,
        output_format='list',
        order_by=['attribute.start_time DESC'],
        max_results=10,
        *args,
        **kwargs,
    )


if __name__ == '__main__':
    main()
