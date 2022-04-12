import argparse
import io
import logging
import pickle
from functools import partial
from typing import Optional

from joblib import Parallel, delayed
from mlflow import set_experiment, start_run, set_tag, log_text
from pandas import DataFrame

from deps.common import get_data_cached
from deps.constants import RANDOM_STATE
# noinspection PyUnresolvedReferences
from deps.pipelines import get_pipelines
from hcve_lib.custom_types import Splits, SplitInput
from hcve_lib.custom_types import Target, SplitPrediction
from hcve_lib.cv import cross_validate
from hcve_lib.feature_importance import get_permutation_importance
from hcve_lib.functional import always
from hcve_lib.splitting import filter_missing_features
from hcve_lib.splitting import get_lco_splits
from hcve_lib.tracking import get_active_experiment_id
from hcve_lib.utils import get_keys
from run_nested_optimization import get_nested_optimization


def get_training_curve_file(method_name: str):
    return f'./data/training_curves_features_{method_name}.data'


def run_training_curves(
    method_name: str,
    optimized: bool,
    n_trials: int,
    n_jobs: int,
    split_name: Optional[str],
    step: int,
    dense_step_threshold: Optional[int],
):
    set_experiment('training_curves')

    with start_run(run_name=method_name) as run:
        data, metadata, X, y = get_data_cached()

        all_splits = get_lco_splits(X, y, data=data)
        if split_name:
            splits = get_keys([split_name], all_splits)
        else:
            splits = all_splits

        if n_jobs != 1:
            results = dict(
                Parallel(n_jobs=len(splits))(
                    (
                        delayed(get_training_curve)(
                            data,
                            X,
                            y,
                            method_name,
                            cohort_name,
                            splits[cohort_name],
                            optimized,
                            step,
                            dense_step_threshold,
                            n_trials,
                            n_jobs,
                            run.info.run_id,
                            get_active_experiment_id(),
                        )
                    ) for cohort_name,
                    split in splits.items()
                )
            )
        else:
            results = dict(
                get_training_curve(
                    data,
                    X,
                    y,
                    method_name,
                    cohort_name,
                    splits[cohort_name],
                    optimized,
                    step,
                    dense_step_threshold,
                    n_trials,
                    n_jobs,
                    run.info.run_id,
                    get_active_experiment_id(),
                ) for cohort_name,
                split in splits.items()
            )

        with open(get_training_curve_file(method_name), 'wb') as f:
            pickle.dump(results, f)


def get_training_curve(
    data: DataFrame,
    X: DataFrame,
    y: Target,
    method_name: str,
    cohort_name: str,
    split: SplitInput,
    optimized: bool,
    step: int,
    dense_step_threshold: Optional[int],
    n_trials: int,
    n_jobs: int,
    parent_run_id: str,
    experiment_id: str,
):
    method = get_pipelines()[method_name]
    X_actual = X.copy()
    results_per_n_features = {}
    importance = None
    current_columns = list(X_actual.columns)

    logger = logging.getLogger('training_curve')
    logger.setLevel(logging.DEBUG)
    log_capture_string = io.StringIO()
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    with start_run(run_id=parent_run_id):
        with start_run(
            run_name=cohort_name,
            nested=True,
            experiment_id=experiment_id,
        ):
            while len(X_actual.columns) >= 1:
                # try:
                splits: Splits = {'tt': split}
                result: SplitPrediction
                n_features = len(X_actual.columns)

                logger.info(n_features)
                set_tag('processing_n_features', n_features)

                if optimized:
                    result = cross_validate(
                        X_actual,
                        y,
                        partial(
                            get_nested_optimization,
                            data=data,
                            method=method,
                            n_trials=n_trials,
                            log=False,
                        ),
                        method.predict,
                        get_splits=always(splits),
                        train_test_filter_callback=filter_missing_features,
                        n_jobs=n_jobs,
                        mlflow_track=False,
                        random_state=RANDOM_STATE,
                    )['tt']
                else:
                    result = cross_validate(
                        X_actual,
                        y,
                        method.get_estimator,
                        method.predict,
                        get_splits=always(splits),
                        n_jobs=-1,
                        train_test_filter_callback=filter_missing_features,
                        random_state=RANDOM_STATE,
                    )['tt']

                new_importance = get_permutation_importance(result, X, y, n_repeats=1, random_state=RANDOM_STATE)
                importance = new_importance.loc[result['X_columns']].sort_values(by=0, ascending=False)
                X_actual = X_actual[result['X_columns']]

                # except (Exception, TypeError) as e:
                #     logging.error(e)

                # if importance:
                #     X_actual = X_actual.drop(columns=importance.index[-1])
                #     continue
                # else:
                #     raise e

                if len(importance) == 0:
                    break

                results_per_n_features[n_features] = result

                actual_step = get_actual_step(dense_step_threshold, importance, step)

                logger.info(f'Dropping {list(importance.iloc[-actual_step:].index)}')

                X_actual = X_actual.drop(columns=list(importance.iloc[-actual_step:].index))

                log_text(log_capture_string.getvalue(), 'log')

        set_tag('processing_n_features', None)
    log_capture_string.close()

    return cohort_name, results_per_n_features


def get_actual_step(dense_threshold, importance, step):
    if dense_threshold is None or len(importance) <= dense_threshold:
        proposed_step = 1
    else:
        proposed_step = step
    if len(importance) > 1:
        actual_step = min(len(importance) - 1, proposed_step)
    else:
        actual_step = min(len(importance), proposed_step)
    return actual_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method-name', type=str)
    parser.add_argument('--optimized', action='store_true')
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--split-name', default=None)
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--dense-step-threshold', default=10)
    args = parser.parse_args()
    run_training_curves(**vars(args))
