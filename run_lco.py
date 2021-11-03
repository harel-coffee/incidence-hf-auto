import argparse
from typing import List

from mlflow import get_experiment_by_name, start_run, log_metrics, set_tracking_uri, set_tag
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis

from common import brier, log_result
from deps.common import get_variables_cached, get_variables
from deps.methods import METHODS_DEFINITIONS, get_pipeline
from hcve_lib.cv import lco_cv
from hcve_lib.evaluation_functions import compute_metrics_folds, c_index, compute_metrics_ci
from hcve_lib.tracking import log_pickled, log_metrics_ci
from hcve_lib.utils import partial2
from hcve_lib.wrapped_sklearn import DFCoxnetSurvivalAnalysis
# noinspection PyUnresolvedReferences
from deps.ignore_warnings import *
from deps.prediction import run_prediction


def run_lco(selected_methods: List[str]):
    set_tracking_uri('http://localhost:5000')
    experiment = get_experiment_by_name('lco')
    data, metadata, X, y = get_variables()
    cv = lco_cv(data.groupby('STUDY'))

    for method_name in selected_methods:
        with start_run(run_name=method_name, experiment_id=experiment.experiment_id):
            set_tag('method_name', method_name)
            current_method = METHODS_DEFINITIONS[method_name]

            result = run_prediction(
                cv,
                X,
                current_method['process_y'](y),
                current_method['get_estimator'],
                current_method['predict'],
                n_jobs=-1,
            )
            log_result(X, y, current_method, result)

            brier_3_years = partial2(brier, kwargs={'time_point': 3 * 365}, name='brier_3_years')
            metrics_folds = compute_metrics_folds(
                result['predictions'], [c_index, brier_3_years], y
            )
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
