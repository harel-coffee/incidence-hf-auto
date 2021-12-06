import logging
import multiprocessing
from functools import partial

from mlflow import start_run, log_metrics, set_tracking_uri, set_tag, set_experiment

from common import brier, log_result
from deps.common import get_variables_cached
# noinspection PyUnresolvedReferences
from deps.ignore_warnings import *
from deps.logger import logger
from hcve_lib.cv import cross_validate
from hcve_lib.splitting import get_lco_splits
from hcve_lib.evaluation_functions import compute_metrics_folds, c_index
from hcve_lib.utils import partial2_args
from hcve_lib.wrapped_sklearn import DFBinMapper
from deps.pipelines import GBHist, LCO_GB_HYPERPARAMETERS


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class Pool2(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


N_BINS = list(range(5, 50, 3))


def run():
    set_tracking_uri('http://localhost:5000')
    list(map(
        run_bin,
        N_BINS,
    ))


def run_bin(n_bins: int) -> None:
    set_experiment('fed_lco_hist')
    data, metadata, X, y = get_variables_cached()
    cv = get_lco_splits(X, y, data)
    with start_run(run_name=f'{n_bins}'):
        logger.setLevel(logging.INFO)
        set_tag('method_name', 'gb_hist')
        current_method = GBHist
        logger.info(f'Bin {n_bins} started')

        transform = DFBinMapper(n_bins=n_bins)
        X_transformed = transform.fit_transform(X, y)
        result = cross_validate(
            X_transformed,
            y,
            partial(current_method.get_estimator, n_bins=n_bins),
            current_method.predict,
            cv,
            n_jobs=-1,
            logger=logger,
            split_hyperparameters=LCO_GB_HYPERPARAMETERS,
        )
        logger.info(f'Bin {n_bins} finished')

        log_result(X, y, current_method, result)

        metrics_folds = compute_metrics_folds(
            result,
            [
                partial2_args(c_index, kwargs={
                    'X': X,
                    'y': y
                }),
                partial2_args(
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
            with start_run(run_name=fold_name, nested=True):
                log_metrics(fold_metrics)


if __name__ == '__main__':
    run()
