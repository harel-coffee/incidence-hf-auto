import logging

from mlflow import set_tracking_uri, set_experiment, start_run, set_tag

from common import log_info
from deps.common import get_data_cached
# noinspection PyUnresolvedReferences
from hcve_lib.custom_types import Method
from deps.logger import logger
from deps.pipelines import XGBClassifierMethod
from hcve_lib.cv import cross_validate, series_to_target
from hcve_lib.evaluation_functions import compute_metrics_ci, roc_auc
from hcve_lib.splitting import get_kfold_splits, filter_missing_features
from hcve_lib.tracking import log_metrics_ci
from hcve_lib.utils import partial2

set_tracking_uri('http://localhost:5000')
logger.setLevel(logging.INFO)
set_experiment('classify_cohort')

current_method = XGBClassifierMethod
method_name = 'xgb'
features_to_remove = ['CREA', 'HAF', 'SOK', 'CI', 'DRK', 'TRT_LIP', 'QRS']

data, metadata, X, _ = get_data_cached()
X = X.drop(columns=features_to_remove)

y = series_to_target(data['STUDY'])

with start_run(run_name=method_name):
    set_tag('features_removed', features_to_remove)

    result = cross_validate(
        X,
        y,
        current_method.get_estimator,
        current_method.predict,
        get_kfold_splits(X),
        n_jobs=-1,
        logger=logger,
        train_test_filter_callback=filter_missing_features,
    )
    log_info(X, current_method, method_name, result)
    metrics_ci = compute_metrics_ci(
        result,
        [partial2(roc_auc, X=X, y=y)],
    )
    log_metrics_ci(metrics_ci, drop_ci=True)
