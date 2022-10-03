import argparse
import logging
import pickle
import warnings
from typing import Dict, Hashable

import toolz
from mlflow import set_tracking_uri

from deps.common import get_data_cached
from deps.constants import RANDOM_STATE
from deps.logger import logger
from deps.metrics import get_metric
from deps.pipelines import CoxNetPipeline
from hcve_lib.custom_types import TrainTestIndex, Result
from hcve_lib.cv import cross_validate
from hcve_lib.functional import always
from hcve_lib.tracking import load_run_results

set_tracking_uri('http://localhost:5000')

HYPERPARAMETER_ANALYSIS_RUNS = dict(
    rsf='b37db8ab201141f18146b21d24787ead',
    stacking='81a829cbb042424a853b106ec626c57b',
    gb='31bbadc068b4413795acbb87e7172126',
    coxnet='b14038beb60641638d5e9107af9049b9',
)


def get_hyperparameters_dependency_file(
    run_id: str,
    metric: str,
):
    return f'./data/hyperparameters_dependency_{run_id}_{metric}.data'


def log_early_stopping():
    ...


def run(
    run_id: str,
    metric: str,
):

    data, metadata, X, y = get_data_cached()

    metric_ = get_metric(metric, y)

    logger.setLevel(logging.INFO)

    results: Result = load_run_results(run_id)

    hyperparameters_dependency = {}

    for split_name, result in results.items():
        print(split_name)
        inner = []
        outer = []
        hyperparameters = []

        for trial in result['model'].study.trials:
            print(trial.number)
            try:
                result_inner = trial.user_attrs['result_split']
            except KeyError:
                continue

            metric_inner = metric_.get_values(result_inner['train_test'], y)[0]

            try:
                model = result_inner['train_test']['model']
            except KeyError as e:
                logger.warning('Skipped trial', e)
                continue

            def get_pipeline(*args, **kwargs):
                return model

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="all coefficients are zero")
                try:
                    splits: Dict[Hashable, TrainTestIndex] = {0: result['split']}
                    result_outer = cross_validate(
                        X,
                        y,
                        get_pipeline,
                        CoxNetPipeline,
                        get_splits=always(splits),
                        logger=logger,
                        random_state=RANDOM_STATE,
                    )

                    metric_outer = metric_.get_values(result_outer[0], y)[0]
                except:
                    metric_outer = None

            inner.append(metric_inner)
            outer.append(metric_outer)
            hyperparameters.append(
                toolz.merge(trial.user_attrs['hyperparameters'], trial.user_attrs['cv_hyperparameters'])
            )

            # TODO
            # hyperparameters.append(
            #     '<br>' + yaml.dump(
            #         map_recursive(
            #             toolz.merge(
            #                 trial.user_attrs['hyperparameters'],
            #                 trial.user_attrs['cv_hyperparameters']
            #             ),
            #             lambda val: round(val, 3) if hasattr(val, '__round__') else val,
            #         )
            #     ).replace('\n', '<br>').replace('\t', '&nbsp;' * 5)
            # )

            hyperparameters_dependency[split_name] = {
                'inner': inner,
                'outer': outer,
                'hyperparameters': hyperparameters,
            }

        with open(get_hyperparameters_dependency_file(run_id, metric), 'wb') as f:
            pickle.dump(hyperparameters_dependency, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-all', action='store_true', default=False)
    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument(
        '--metric',
        type=str,
    )
    args = parser.parse_args()

    if args.run_all:
        for method, run_id in HYPERPARAMETER_ANALYSIS_RUNS.items():
            run(run_id=run_id, metric=args.metric)
