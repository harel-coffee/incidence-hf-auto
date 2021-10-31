import argparse
from typing import List

from mlflow import get_experiment_by_name, start_run, log_metrics, set_tracking_uri, set_tag
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis

from common import brier
from deps.common import get_variables_cached
from hcve_lib.cv import lm_cv
from hcve_lib.evaluation_functions import c_index, get_2_level_groups, compute_metric_fold
from hcve_lib.tracking import log_pickled
from hcve_lib.utils import transpose_dict, partial2
from hcve_lib.wrapped_sklearn import DFCoxnetSurvivalAnalysis
# noinspection PyUnresolvedReferences
from deps.ignore_warnings import *
from deps.methods import get_pipeline
from deps.prediction import run_prediction


def run_lco(selected_methods: List[str]):
    set_tracking_uri('http://localhost:5000')
    experiment = get_experiment_by_name('lm')
    data, metadata, X, y = get_variables_cached()
    cv = lm_cv(data.groupby('STUDY'))

    methods = {
        'coxnet': lambda: DFCoxnetSurvivalAnalysis(fit_baseline_model=True),
        'gb': lambda: GradientBoostingSurvivalAnalysis(verbose=3),
        'rsf': lambda: RandomSurvivalForest(verbose=3, n_jobs=-1),
    }

    for method_name in selected_methods:
        result = run_prediction(
            cv,
            X,
            y,
            lambda: get_pipeline(methods[method_name](), X),
        )
        metrics_group = transpose_dict(
            get_2_level_groups(result['predictions'], data.groupby('STUDY'))
        )
        for test_fold_name, train_folds in metrics_group.items():
            with start_run(
                run_name=f'{method_name} {test_fold_name}', experiment_id=experiment.experiment_id
            ):
                for train_fold_name, fold in train_folds.items():
                    set_tag('method_name', method_name)
                    if len(fold['y_score']) == 0:
                        continue
                    with start_run(
                        run_name=f'train: {train_fold_name}',
                        experiment_id=experiment.experiment_id,
                        nested=True
                    ):
                        metrics = compute_metric_fold(
                            [c_index, partial2(brier, time_point=365 * 3)], fold
                        )
                        log_metrics(metrics)
                    log_pickled(fold, 'fold')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('selected_methods', metavar='METHOD', type=str, nargs='+')
    args = parser.parse_args()
    run_lco(**vars(args))
