\import argparse
from typing import List

from mlflow import get_experiment_by_name, start_run, set_tracking_uri, set_tag

from deps.common import get_data
# noinspection PyUnresolvedReferences
from deps.ignore_warnings import *
from deps.prediction import run_prediction
from hcve_lib.splitting import get_lm_splits
from hcve_lib.tracking import log_pickled
from deps.pipelines import get_pipelines


def run_lco(selected_methods: List[str]):
    set_tracking_uri('http://localhost:5000')
    experiment = get_experiment_by_name('lm')
    data, metadata, X, y = get_data()
    cv = get_lm_splits(data.groupby('STUDY'))

    for method_name in selected_methods:
        current_method = get_pipelines()[method_name]
        result = run_prediction(
            cv,
            X,
            y,
            current_method.get_estimator,
            current_method.predict,
        )
        with start_run(run_name=f'{method_name}', experiment_id=experiment.experiment_id):
            log_pickled(result, 'result_split')
            set_tag('method_name', method_name)
        # metrics_group = transpose_dict(get_2_level_groups(result_split, data.groupby('STUDY'), data))
        # for test_fold_name, train_folds in metrics_group.items():
        #     with start_run(
        #         run_name=f'{method_name} {test_fold_name}', experiment_id=experiment.experiment_id
        #     ):
        #         for train_fold_name, prediction in train_folds.items():
        #             if prediction is None:
        #                 continue
        #             set_tag('method_name', method_name)
        #
        #             metrics = [
        #                 partial2_args(c_index, kwargs={
        #                     'X': X,
        #                     'y': y
        #                 }),
        #                 partial2_args(brier, kwargs={
        #                     'time_point': 365 * 3,
        #                     'X': X,
        #                     'y': y
        #                 })
        #             ]
        #
        #             with start_run(
        #                 run_name=f'train: {train_fold_name}',
        #                 experiment_id=experiment.experiment_id,
        #                 nested=True
        #             ):
        #                 metrics = compute_metrics_fold(
        #                     metrics,
        #                     prediction,
        #                 )
        #                 log_metrics(metrics)
        #                 log_pickled(prediction, 'prediction')
        #


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('selected_methods', metavar='METHOD', type=str, nargs='+')
    args = parser.parse_args()
    run_lco(**vars(args))
