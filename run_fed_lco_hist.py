import argparse
from functools import partial
from typing import List

from mlflow import get_experiment_by_name, start_run, log_metrics, set_tracking_uri, set_tag

from common import brier, log_result
from deps.common import get_variables_cached
# noinspection PyUnresolvedReferences
from deps.ignore_warnings import *
from deps.logger import logger
from hcve_lib.cv import get_lco_splits, cross_validate
from hcve_lib.evaluation_functions import compute_metrics_folds, c_index
from hcve_lib.utils import partial2
from pipelines import GBHist, LCO_GB_HYPERPARAMETERS


def run():
    set_tracking_uri('http://localhost:5000')
    experiment = get_experiment_by_name('fed_lco_hist')
    data, metadata, X, y = get_variables_cached()
    cv = get_lco_splits(X, y, data)

    for n_bins in range(5, 100, 10):
        n_bins = 100
        with start_run(run_name=f'{n_bins}', experiment_id=experiment.experiment_id):
            logger.setLevel('DEBUG')
            set_tag('method_name', 'gb_hist')
            current_method = GBHist

            result = cross_validate(
                X,
                y,
                partial(current_method.get_estimator, n_bins=n_bins),
                current_method.predict,
                cv,
                n_jobs=-1,
                logger=logger,
                split_hyperparameters=LCO_GB_HYPERPARAMETERS,
            )

            log_result(X, y, current_method, result)

            metrics_folds = compute_metrics_folds(
                result,
                [
                    partial2(c_index, kwargs={
                        'X': X,
                        'y': y
                    }),
                    partial2(
                        brier,
                        kwargs={
                            'X': X,
                            'y': y,
                            'time_point': 3 * 365,
                        },
                        name='brier_3_years',
                    ),
                ],
            )

            for fold_name, fold_metrics in metrics_folds.items():
                with start_run(
                    run_name=fold_name, nested=True, experiment_id=experiment.experiment_id
                ):
                    log_metrics(fold_metrics)

        return


if __name__ == '__main__':
    run()
