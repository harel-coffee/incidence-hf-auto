from mlflow import get_experiment_by_name, start_run, log_metrics
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis

from deps.data import load_metadata, load_data_cached
from hcve_lib.cv import lco_cv, kfold_cv
from hcve_lib.data import to_survival_y_records, get_X, get_survival_y
from hcve_lib.evaluation_functions import compute_metrics_folds, c_index, compute_metrics_ci
from hcve_lib.functional import pipe
from hcve_lib.tracking import log_pickled, log_metrics_ci
from hcve_lib.wrapped_sklearn import DFCoxnetSurvivalAnalysis
# noinspection PyUnresolvedReferences
from ignore_warnings import *
from pipelines import get_pipeline
from prediction import run

experiment = get_experiment_by_name('10_fold')
metadata = load_metadata()
data = load_data_cached(metadata)
X, y = pipe(
    (
        get_X(data, metadata),
        to_survival_y_records(get_survival_y(data, 'NFHF', metadata)),
    ),
)
cv = kfold_cv(data, n_splits=10, shuffle=True, random_state=243)
methods = {
    'coxnet': DFCoxnetSurvivalAnalysis(),
    'gb': GradientBoostingSurvivalAnalysis(verbose=3),
    'rsf': RandomSurvivalForest(verbose=3),
}

for estimator_name, estimator in methods.items():
    with start_run(run_name=estimator_name, experiment_id=experiment.experiment_id):
        result = run(
            cv,
            data,
            metadata,
            lambda: get_pipeline(estimator, X),
        )

        log_pickled(result, 'result')
        metrics_ci = compute_metrics_ci(result['predictions'], [c_index])
        log_metrics_ci(metrics_ci)

        metrics_folds = compute_metrics_folds(result['predictions'], [c_index])
        for fold_name, fold_metrics in metrics_folds.items():
            with start_run(run_name=fold_name, nested=True, experiment_id=experiment.experiment_id):
                log_metrics(fold_metrics)

# for name, metrics_fold in metrics_folds.items():
#     with start_method_run(name):
#
