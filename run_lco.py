import argparse
from typing import List

from mlflow import get_experiment_by_name, start_run, log_metrics, set_tracking_uri, set_tag
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis

from deps.common import get_variables_cached
from deps.methods import METHODS_DEFINITIONS, get_pipeline
from hcve_lib.cv import lco_cv
from hcve_lib.evaluation_functions import compute_metrics_folds, c_index, compute_metrics_ci
from hcve_lib.tracking import log_pickled, log_metrics_ci
from hcve_lib.wrapped_sklearn import DFCoxnetSurvivalAnalysis
# noinspection PyUnresolvedReferences
from deps.ignore_warnings import *
from deps.prediction import run_prediction


def run_lco(selected_methods: List[str]):
    set_tracking_uri('http://localhost:5000')
    experiment = get_experiment_by_name('lco')
    data, metadata, X, y = get_variables_cached()
    cv = lco_cv(data.groupby('STUDY'))

    for method_name in selected_methods:
        with start_run(run_name=method_name, experiment_id=experiment.experiment_id):
            set_tag('method_name', method_name)

            result = run_prediction(
                cv,
                X,
                y,
                lambda: get_pipeline(METHODS_DEFINITIONS[method_name]['get_estimator'](), X),
            )
            metrics_ci = compute_metrics_ci(result['predictions'], [c_index])
            log_pickled(result, 'result')
            log_metrics_ci(metrics_ci)
            metrics_folds = compute_metrics_folds(result['predictions'], [c_index])
            for fold_name, fold_metrics in metrics_folds.items():
                with start_run(
                    run_name=fold_name, nested=True, experiment_id=experiment.experiment_id
                ):
                    log_metrics(fold_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('selected_methods', metavar='METHOD', type=str, nargs='+')
    args = parser.parse_args()
    run_lco(**vars(args))
