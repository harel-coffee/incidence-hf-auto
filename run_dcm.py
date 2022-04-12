from mlflow import set_tracking_uri, start_run, get_experiment_by_name

from deps.common import get_data_cached
from deps.logger import logger
from hcve_lib.cv import cross_validate
from hcve_lib.splitting import get_lco_splits, filter_missing_features
from hcve_lib.evaluation_functions import compute_metrics_on_splits, c_index_inverse_score, brier_y_score
from hcve_lib.utils import partial2_args
from deps.pipelines import DeepCoxMixtureMethod

data, metadata, X, y = get_data_cached()

k = 1
h = 20

set_tracking_uri('http://localhost:5000')
experiment = get_experiment_by_name('default_reproduce')
with start_run(run_name='dcm', experiment_id=experiment.experiment_id):
    # predictions = train_test_filter(
    #     data,
    #     train_filter=lambda _data: _data['STUDY'].isin(['HEALTHABC', 'PREDICTOR', 'PROSPER']),
    #     test_filter=lambda _data: _data['STUDY'] == 'ASCOT'
    # )
    splits = get_lco_splits(X, data)

    result = cross_validate(
        X,
        y,
        DeepCoxMixtureMethod.get_estimator,
        DeepCoxMixtureMethod.predict,
        splits,
        n_jobs=-1,
        logger=logger,
        train_test_filter_callback=filter_missing_features,
    )
    # log_result(X, y, DeepCoxMixtureMethod, result_split)
    c_index_ = partial2_args(c_index_inverse_score, kwargs={'X': X, 'y': y})
    brier_3_years = partial2_args(brier_y_score, kwargs={'time_point': 3 * 365, 'X': X, 'y': y})
    metrics_ci = compute_metrics_on_splits(
        result,
        [brier_3_years, c_index_],
        y,
    )
    print(metrics_ci)
    # log_metrics_ci(metrics_ci)
    # print(metrics_ci)
