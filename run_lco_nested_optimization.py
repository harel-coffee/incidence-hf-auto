import argparse
from functools import partial
from logging import INFO
from typing import List

import optuna
from mlflow import get_experiment_by_name, start_run, set_tracking_uri, log_text, set_tag, log_metrics, active_run
from pandas import DataFrame

from common import log_result, brier
from deps.common import get_variables
# noinspection PyUnresolvedReferences
from deps.custom_types import MethodDefinition
from deps.logger import logger
from hcve_lib.context import get_context
from hcve_lib.custom_types import FoldPrediction
from hcve_lib.cv import Optimize, train_test_proportion, cross_validate, filter_missing_features, OptimizeEstimator
from hcve_lib.cv import lco_cv
from hcve_lib.data import to_survival_y_records
from hcve_lib.evaluation_functions import c_index, compute_metrics_folds
from hcve_lib.optimization import optuna_report_mlflow, EarlyStoppingCallback
from hcve_lib.tracking import get_active_experiment_id
from hcve_lib.utils import partial2
from pipelines import get_pipelines

study = optuna.create_study(direction="maximize")


def run(methods: List[str]) -> None:
    data, metadata, X, y = get_variables()
    logger.setLevel(INFO)
    logger.info('Data loaded')
    cv = lco_cv(data.groupby('STUDY'))
    set_tracking_uri('http://localhost:5000')
    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    experiment = get_experiment_by_name('lco_nested')
    for method_name in methods:
        method_definition = get_pipelines()[method_name]
        with start_run(run_name=method_name, experiment_id=experiment.experiment_id):
            set_tag("method_name", method_name)
            set_tag("nested", True)
            result = cross_validate(
                X,
                y,
                lambda _: get_optimize(data, method_definition=method_definition),
                method_definition.predict,
                cv,
                train_test_filter_callback=filter_missing_features,
                n_jobs=-1,
                mlflow_track=True,
            )

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


def nested_optuna_mlflow(study, trial):
    return optuna_report_mlflow(study, trial)


def get_optimize(data: DataFrame, method_definition: MethodDefinition):
    return OptimizeEstimator(
        partial(method_definition.get_estimator, verbose=True),
        method_definition.optuna,
        scoring,
        method_definition.predict,
        get_cv=partial(get_lco_cv, data=data),
        optimize_params={
            'n_jobs': -1,
            'n_trials': 1,
        },
        optimize_callbacks=[
            nested_optuna_mlflow,
            EarlyStoppingCallback(
                early_stopping_rounds=50,
                direction='maximize',
                stop_callback=log_early_stopping,
            )
        ],
        logger=logger,
    )


def get_lco_cv(X, _, data):
    return lco_cv(data.loc[X.index].groupby("STUDY"))


def scoring(estimator, X_test, y_true):
    y_score = estimator.predict(X_test)
    return c_index(FoldPrediction(y_true=y_true, y_score=y_score, model=estimator))


def log_early_stopping(it):
    logger.info(f'Early stopping after {it} iterations')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('methods', metavar='METHOD', type=str, nargs='+')
    args = parser.parse_args()
    run(**vars(args))
