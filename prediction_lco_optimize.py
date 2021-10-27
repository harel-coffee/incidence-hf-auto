from hcve_lib.cv import lco_cv
from hcve_lib.wrapped_sklearn import DFCoxnetSurvivalAnalysis
from mlflow import get_experiment_by_name
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis

from common import get_variables_cached
# noinspection PyUnresolvedReferences
from ignore_warnings import *
from hcve_lib.cv import Optimize

from pipelines import get_pipeline

experiment = get_experiment_by_name('lco_optimized')

data, metadata, X, y = get_variables_cached()

cv = lco_cv(data.groupby('STUDY'))

methods = {
    'coxnet': DFCoxnetSurvivalAnalysis(),
    'gb': GradientBoostingSurvivalAnalysis(verbose=3),
    'rsf': RandomSurvivalForest(verbose=3),
}

print(cv['ASCOT'][0])
Optimize(get_pipeline(estimator, X))
#
# #
# for estimator_name, estimator in methods.items():
#     with start_run(run_name=estimator_name, experiment_id=experiment.experiment_id):
#         result = run(
#             cv,
#             data,
#             metadata,
#             lambda: get_pipeline(estimator, X),
#         )
#
#         log_pickled(result, 'result')
#         metrics_ci = compute_metrics_ci(result['predictions'], [c_index])
#         log_metrics_ci(metrics_ci)
#
#         metrics_folds = compute_metrics_folds(result['predictions'], [c_index])
#         for fold_name, fold_metrics in metrics_folds.items():
#             with start_run(run_name=fold_name, nested=True, experiment_id=experiment.experiment_id):
#                 log_metrics(fold_metrics)

# for name, metrics_fold in metrics_folds.items():
#     with start_method_run(name):
#
