import argparse
from typing import List

from mlflow import get_experiment_by_name, start_run, log_metrics, set_tracking_uri, set_tag

from common import log_result
from deps.common import get_variables_cached
from deps.logger import logger
from hcve_lib.cv import get_lco_splits, cross_validate
# noinspection PyUnresolvedReferences
from deps.ignore_warnings import *
from pipelines import get_pipelines
from utils import compute_standard_metrics


def run_lco(selected_methods: List[str]):
    set_tracking_uri('http://localhost:5000')
    experiment = get_experiment_by_name('lco')
    data, metadata, X, y = get_variables_cached()
    cv = get_lco_splits(X, y, data)

    for method_name in selected_methods:
        with start_run(run_name=method_name, experiment_id=experiment.experiment_id):
            current_method = get_pipelines()[method_name]
            logger.setLevel('DEBUG')

            result = cross_validate(
                X,
                y,
                current_method.get_estimator,
                current_method.predict,
                cv,
                n_jobs=-1,
                logger=logger,
            )

            log_result(X, y, current_method, method_name, result)

            metrics_folds = compute_standard_metrics(X, y, result)

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
