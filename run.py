import logging
import pickle
import warnings
from typing import Callable, Any

import matplotlib
import mlflow
import numpy as np
import pandas
from matplotlib import pyplot
from mlflow.utils.file_utils import TempDir
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier, StackingClassifier
from sklearn.model_selection import KFold, RepeatedKFold
from xgboost import XGBClassifier

from hcve_lib.custom_types import FoldPrediction
from hcve_lib.cv import cross_validate
from hcve_lib.evaluation_functions import compute_classification_metrics_from_result, \
    compute_classification_metrics_fold, get_1_class_y_score
from hcve_lib.statistics import confidence_interval
from hcve_lib.wrapped_sklearn import DFLogisticRegression, DFPipeline, DFSimpleImputer, DFStandardScaler

RANDOM_STATE = 4651321
N_REPEATS = 50

logging.basicConfig(
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)


def main():
    logger = logging.getLogger('run')
    logger.setLevel(logging.INFO)
    pyplot.style.use('dark_background')

    matplotlib.rcParams['figure.dpi'] = 200

    X_eps, y_eps = get_eps_X_y()
    X_psk, y_psk = get_psk_X_y()

    METHODS = {
        'logistic regression': lambda: make_pipeline(DFLogisticRegression(class_weight='balanced')),
        'random forest': lambda: make_pipeline(RandomForestClassifier(class_weight='balanced')),
        'adaboost': lambda: make_pipeline(AdaBoostClassifier()),
        'gradient boosting': lambda: make_pipeline(GradientBoostingClassifier()),
        'extra tree classifier': lambda:
        make_pipeline(ExtraTreesClassifier(class_weight='balanced')),
        'xgboost': lambda: make_pipeline(
            XGBClassifier(
                max_bin=100,
                tree_method='hist',
                use_label_encoder=False,
                eval_metric='logloss',
            )
        ),
        # 'xgboost_balanced':
        # lambda: BalancedXGBoostClassifier(),
        # 'voting classifier': lambda: VotingClassifier(),
        # 'random survival forest': lambda RandomSurvivalForest(),
        # TPOTClassifier,
    }

    stacking_method = StackingClassifier(
        [(_method_name, _get_method()) for _method_name, _get_method in METHODS.items()]
    )

    # METHODS['stacking classifier'] = lambda: stacking_method

    for method_name, get_method in METHODS.items():
        with mlflow.start_run(run_name=method_name, experiment_id='1'):
            logger.info(f'Method {method_name}')

            mlflow.log_param('method', method_name)

            # with mlflow.start_run(run_name='cv_eps', nested=True):
            #     logger.info(f'\t- CV EPS')
            #     run_cv(X_eps, y_eps, get_pipeline_method)

            # with mlflow.start_run(run_name='cv_psk', nested=True):
            #     logger.info(f'\t- CV PSK')
            #     run_cv(X_psk, y_psk, get_pipeline_method)
            #
            # with mlflow.start_run(run_name='cv_merged', nested=True):
            #     logger.info(f'\t- CV MERGED')
            #     run_cv(X_merged, y_merged, get_pipeline_method)
            #
            with start_method_run('cross cv psk -> eps', method_name, logger):
                run_cross_cv(
                    X_psk,
                    y_psk,
                    X_eps,
                    y_eps,
                    get_method,
                    (
                        lambda pipeline, eps_train, eps_test, psk_train, psk_test: pipeline.fit(
                            X_psk.iloc[psk_train],
                            y_psk.iloc[psk_train],
                        )
                    ),
                    (
                        lambda pipeline, eps_train, eps_test, psk_train, psk_test:
                        test_and_get_metrics(
                            pipeline,
                            X_eps.iloc[eps_test].copy(),
                            y_eps.iloc[eps_test].copy(),
                        )
                    ),
                ),

            with start_method_run('cross cv eps -> psk', method_name, logger):
                run_cross_cv(
                    X_psk,
                    y_psk,
                    X_eps,
                    y_eps,
                    get_method,
                    (
                        lambda pipeline, eps_train, eps_test, psk_train, psk_test: pipeline.fit(
                            X_eps.iloc[eps_train],
                            y_eps.iloc[eps_train],
                        )
                    ),
                    (
                        lambda pipeline, eps_train, eps_test, psk_train, psk_test:
                        test_and_get_metrics(
                            pipeline,
                            X_psk.iloc[psk_test].copy(),
                            y_psk.iloc[psk_test].copy(),
                        )
                    ),
                ),

            with start_method_run(
                'cross cv psk ∪ eps  -> eps',
                method_name,
                logger,
            ):
                run_cross_cv(
                    X_psk,
                    y_psk,
                    X_eps,
                    y_eps,
                    get_method,
                    (
                        lambda pipeline, eps_train, eps_test, psk_train, psk_test: pipeline.fit(
                            pandas.concat([
                                X_psk.iloc[psk_train],
                                X_eps.iloc[eps_train],
                            ]),
                            pandas.concat([
                                y_psk.iloc[psk_train],
                                y_eps.iloc[eps_train],
                            ]),
                        )
                    ),
                    (
                        lambda pipeline, eps_train, eps_test, psk_train, psk_test:
                        test_and_get_metrics(
                            pipeline,
                            X_eps.iloc[eps_test].copy(),
                            y_eps.iloc[eps_test].copy(),
                        )
                    ),
                ),

                with start_method_run(
                    'cross cv psk ∪ eps  -> psk',
                    method_name,
                    logger,
                ):
                    run_cross_cv(
                        X_psk,
                        y_psk,
                        X_eps,
                        y_eps,
                        get_method,
                        (
                            lambda pipeline, eps_train, eps_test, psk_train, psk_test: pipeline.fit(
                                pandas.concat([
                                    X_psk.iloc[psk_train],
                                    X_eps.iloc[eps_train],
                                ]),
                                pandas.concat([
                                    y_psk.iloc[psk_train],
                                    y_eps.iloc[eps_train],
                                ]),
                            )
                        ),
                        (
                            lambda pipeline, eps_train, eps_test, psk_train, psk_test:
                            test_and_get_metrics(
                                pipeline,
                                X_psk.iloc[psk_test].copy(),
                                y_psk.iloc[psk_test].copy(),
                            )
                        ),
                    ),

            # sub_run(
            #     cross_validate_predict,
            #     'cv eps',
            #     logger,
            #     lambda: run_cv(X_eps, y_eps, get_pipeline_method),
            # )
            #
            # sub_run(
            #     cross_validate_predict,
            #     'cv psk',
            #     logger,
            #     lambda: run_cv(X_psk, y_psk, get_pipeline_method),
            # )
            #
            # sub_run(
            #     cross_validate_predict, 'psk -> eps', logger,
            #     lambda: run_external_validation(
            #         X_psk_train,
            #         y_psk_train,
            #         X_eps_test,
            #         y_eps_test,
            #         get_pipeline_method,
            #     ))
            #
            # sub_run(
            #     cross_validate_predict, 'eps -> psk', logger,
            #     lambda: run_external_validation(
            #         X_eps_train,
            #         y_eps_train,
            #         X_psk_test,
            #         y_psk_test,
            #         get_pipeline_method,
            #     ))

            # with mlflow.start_run(run_name='psk -> eps', nested=True):
            #     logger.info(f'\t- PSK -> EPS')
            #     run_external_validation(
            #         X_psk,
            #         y_psk,
            #         X_eps,
            #         y_eps,
            #         get_pipeline_method,
            #     )


def sub_run(
    method_name: str,
    run_name: str,
    logger: logging.Logger,
    callback: Callable,
) -> None:
    with start_method_run(run_name, method_name, logger):
        callback()


def start_method_run(
    run_name: str,
    method_name: str,
    logger: logging.Logger,
) -> mlflow.ActiveRun:
    logger.info(f'\t- {run_name}')
    run = mlflow.start_run(run_name=run_name, nested=True, experiment_id='1')
    mlflow.log_param('run_name', run_name)
    mlflow.log_param('method', method_name)
    return run


def run_external_validation(X_train, y_train, X_test, y_test, _get_pipeline):
    metrics, model = run_train_test(
        _get_pipeline,
        X_train,
        X_test,
        y_test,
        y_train,
    )

    log_model(model)
    print(metrics['roc_auc'])
    log_metrics(metrics)


def run_train_test(get_pipeline, X_train, X_test, y_test, y_train):
    pipeline = get_pipeline()
    pipeline.fit(X_train, y_train)
    metrics = test_and_get_metrics(pipeline, X_test, y_test)
    return metrics, pipeline


def test_and_get_metrics(pipeline, X_test, y_test):
    y_score = get_1_class_y_score(
        DataFrame(
            pipeline.predict_proba(X_test),
            index=X_test.index_frame,
        )
    )
    result = FoldPrediction(y_true=y_test, y_score=y_score)
    metrics = compute_classification_metrics_fold(
        result,
        ignore_warning=True,
    )
    return metrics


def run_cross_cv(
    X_psk: DataFrame,
    y_psk: Series,
    X_eps: DataFrame,
    y_eps: Series,
    _get_pipeline: Callable,
    train_callback: Callable,
    test_callback: Callable,
) -> None:
    warnings.filterwarnings(action='ignore', category=UserWarning)

    eps_splits = list(
        RepeatedKFold(
            n_splits=10,
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE,
        ).split(X_eps, y_eps)
    )

    psk_splits = list(
        RepeatedKFold(
            n_splits=10,
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE,
        ).split(X_psk, y_psk)
    )

    aucs = []

    for (
        eps_train,
        eps_test,
    ), (
        psk_train,
        psk_test,
    ) in zip(eps_splits, psk_splits):
        pipeline = _get_pipeline()

        train_callback(
            pipeline,
            eps_train,
            eps_test,
            psk_train,
            psk_test,
        )

        metrics = test_callback(
            pipeline,
            eps_train,
            eps_test,
            psk_train,
            psk_test,
        )

        # pipeline_eps = _get_pipeline()
        #
        # pipeline_eps.fit(
        #     X_eps.iloc[eps_train],
        #     y_eps.iloc[eps_train],
        # )
        #
        # metrics_psk = run_test(
        #     pipeline_eps,
        #     X_psk.iloc[psk_test].copy(),
        #     y_psk.iloc[psk_test].copy(),
        # )

        aucs.append(metrics['roc_auc'])
        # psk_roc_aucs.append(metrics_psk['roc_auc'])

    mlflow.log_metric('roc_auc', float(np.mean(aucs)))

    ci_values = confidence_interval(aucs)[1]
    mlflow.log_metric('roc_auc_l', ci_values[0])
    mlflow.log_metric('roc_auc_r', ci_values[1])


def run_cv(X, y, _get_pipeline):
    result = cross_validate(
        X,
        y,
        _get_pipeline,
        KFold(n_splits=10).split,
        report_batch=lambda models, *_: log_model(models),
        n_jobs=1
    )

    metrics = compute_classification_metrics_from_result(
        result['scores'],
        ignore_warning=True,
    )
    log_metrics(metrics)


def log_metrics(metrics):
    for name, value in metrics.items():

        try:
            numerical_value = value['mean']
        except (IndexError, TypeError):
            numerical_value = value

        mlflow.log_metric(name, numerical_value)


def make_pipeline(estimator):
    return DFPipeline(
        [
            ('imputer', DFSimpleImputer()),
            ('scaler', DFStandardScaler()),
            ('method', estimator),
        ]
    )


def log_model(model: Any) -> None:
    log_pickled(model, 'model')


def log_pickled(pipeline: Any, path: str) -> None:
    with TempDir() as tmp_dir:
        path = tmp_dir.path() + '/' + path
        with open(path, 'bw') as file:
            pickle.dump(pipeline, file, protocol=5)
        mlflow.log_artifact(path)


# for cross_validate_predict, get_method in METHODS.items():
#     with mlflow.start_run(run_name=cross_validate_predict):
#         result = cross_validate(
#             X,
#             y,
#             KFold(n_splits=10).split,
#             lambda: Pipeline([('imputer', SimpleImputer()),
#                               ('scaler', StandardScaler()),
#                               ('method', get_method())]),
#         )
#         mlflow.set_tags({
#             'federated': False,
#             'method': cross_validate_predict,
#         })
#         metrics = compute_classification_metrics_from_result(result['scores'])
#         for name, value in metrics.items():
#             mlflow.log_metric(name, value['mean'])
#
#         print(metrics)

if __name__ == '__main__':
    main()
