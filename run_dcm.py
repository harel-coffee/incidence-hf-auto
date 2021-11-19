from mlflow import set_tracking_uri, start_run, get_experiment_by_name

from common import brier_y_score
from deps.common import get_variables_cached
from deps.logger import logger
from hcve_lib.cv import cross_validate, get_lco_splits
from hcve_lib.evaluation_functions import compute_metrics_folds, c_index_inverse_score
from hcve_lib.utils import partial2
from pipelines import DeepCoxMixtureMethod

data, metadata, X, y = get_variables_cached()

k = 1
h = 20

set_tracking_uri('http://localhost:5000')
experiment = get_experiment_by_name('reproduce')
with start_run(run_name='dcm', experiment_id=experiment.experiment_id):
    # splits = train_test_filter(
    #     data,
    #     train_filter=lambda _data: _data['STUDY'].isin(['HEALTHABC', 'PREDICTOR', 'PROSPER']),
    #     test_filter=lambda _data: _data['STUDY'] == 'ASCOT'
    # )
    splits = get_lco_splits(data.groupby('STUDY'))

    result = cross_validate(
        X,
        y,
        DeepCoxMixtureMethod.get_estimator,
        DeepCoxMixtureMethod.predict,
        splits,
        n_jobs=-1,
        logger=logger,
    )
    # log_result(X, y, DeepCoxMixtureMethod, result)
    c_index_ = partial2(c_index_inverse_score, kwargs={'X': X, 'y': y})
    brier_3_years = partial2(brier_y_score, kwargs={'time_point': 3 * 365, 'X': X, 'y': y})
    metrics_ci = compute_metrics_folds(
        result,
        [brier_3_years, c_index_],
    )
    print(metrics_ci)
    # log_metrics_ci(metrics_ci)
    # print(metrics_ci)
