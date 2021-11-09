import argparse
from functools import partial
from typing import List

from mlflow import get_experiment_by_name, start_run, log_metrics, set_tracking_uri, set_tag

from common import brier
from deps.common import get_variables
# noinspection PyUnresolvedReferences
from deps.ignore_warnings import *
from deps.prediction import run_prediction
from hcve_lib.cv import lm_cv
from hcve_lib.evaluation_functions import c_index, get_2_level_groups, compute_metric_fold
from hcve_lib.tracking import log_pickled
from hcve_lib.utils import transpose_dict, partial2
from pipelines import get_pipelines


def run_lco(selected_methods: List[str]):
    set_tracking_uri('http://localhost:5000')
    experiment = get_experiment_by_name('lm')
    data, metadata, X, y = get_variables()
    cv = lm_cv(data.groupby('STUDY'))

    for method_name in selected_methods:
        current_method = get_pipelines()[method_name]
        result = run_prediction(
            cv,
            X,
            current_method.process_y(y),
            current_method.get_estimator,
            current_method.predict,
        )
        metrics_group = transpose_dict(get_2_level_groups(result, data.groupby('STUDY'), data))
        for test_fold_name, train_folds in metrics_group.items():
            with start_run(
                run_name=f'{method_name} {test_fold_name}', experiment_id=experiment.experiment_id
            ):
                for train_fold_name, fold in train_folds.items():
                    if fold is None:
                        continue
                    set_tag('method_name', method_name)
                    with start_run(
                        run_name=f'train: {train_fold_name}',
                        experiment_id=experiment.experiment_id,
                        nested=True
                    ):
                        metrics = compute_metric_fold(
                            [
                                partial2(c_index, kwargs={
                                    'X': X,
                                    'y': y
                                }),
                                partial2(brier, kwargs={
                                    'time_point': 365 * 3,
                                    'X': X,
                                    'y': y
                                })
                            ],
                            fold,
                        )
                        log_metrics(metrics)
                        log_pickled(fold, 'fold')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('selected_methods', metavar='METHOD', type=str, nargs='+')
    args = parser.parse_args()
    run_lco(**vars(args))
