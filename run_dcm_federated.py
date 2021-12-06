from mlflow import set_tracking_uri, start_run, get_experiment_by_name

from common import brier_y_score
from deps.common import get_variables_cached
from deps.logger import logger
from hcve_lib.cv import cross_validate
from hcve_lib.splitting import get_reproduce_split
from hcve_lib.evaluation_functions import compute_metrics_folds, c_index
from hcve_lib.utils import partial2_args
from deps.pipelines import DeepCoxMixtureMethod

data, metadata, X, y = get_variables_cached()

k = 1
h = 20

set_tracking_uri('http://localhost:5000')
experiment = get_experiment_by_name('federated')
with start_run(run_name='dcm', experiment_id=experiment.experiment_id):
    splits = get_reproduce_split(X, y, data)

    result = cross_validate(
        X,
        y,
        DeepCoxMixtureMethod.get_estimator,
        DeepCoxMixtureMethod.predict,
        splits,
        n_jobs=1,
        logger=logger,
        split_hyperparameters={'train_test_filter': {
            'estimator': {
                'k': 4,
                'layers': [42]
            }
        }},
    )

    # log_result(X, y, DeepCoxMixtureMethod, result_slit)
    c_index_ = partial2_args(c_index, kwargs={'X': X, 'y': y})
    brier_3_years = partial2_args(brier_y_score, kwargs={'time_point': 3 * 365, 'X': X, 'y': y})
    metrics_ci = compute_metrics_folds(
        result,
        [brier_3_years, c_index_],
    )
    print(metrics_ci)
    # log_metrics_ci(metrics_ci)
    # print(metrics_ci)
