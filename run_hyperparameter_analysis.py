import argparse
import logging
import pickle
import warnings
from copy import copy
from typing import Dict, Hashable

import toolz
import yaml
from mlflow import set_tracking_uri

from deps.common import get_data_cached
from deps.logger import logger
from deps.pipelines import CoxNet
from hcve_lib.custom_types import SplitPrediction, SplitInput
from hcve_lib.cv import cross_validate
from hcve_lib.evaluation_functions import compute_metrics_on_splits, c_index
from hcve_lib.tracking import load_run_results
from hcve_lib.utils import map_recursive
from hcve_lib.utils import partial2

set_tracking_uri('http://localhost:5000')

nested_lm_coxnet_id = '5c5fe8a7427b49e0bd1e8a07e391389e'
nested_lco_gb_id = 'babd869f42934e92b912c53d81c3664c'


def get_hyperparameters_dependency_file(run_id: str):
    return f'./data/hyperparameters_dependency_{run_id}.data'


def log_early_stopping():
    ...


def run(run_id: str):
    logger.setLevel(logging.INFO)

    results: Result = load_run_results(run_id)

    data, metadata, X, y = get_data_cached()

    hyperparameters_dependency = {}

    for split_name, result in results.items():
        print(split_name)
        c_indexes_inner = []
        c_indexes_outer = []
        hyperparameters = []

        for trial in result['model'].study.trials:
            print(trial.number)
            try:
                result_inner = trial.user_attrs['result_split']
            except KeyError:
                continue

            c_index_inner = compute_metrics_on_splits(
                result_inner,
                [
                    partial2(c_index, X=X, y=y),
                ],
                y,
            )['train_test']['c_index']

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
                    splits: Dict[Hashable, SplitInput] = {0: result['split']}
                    result_outer = cross_validate(
                        X,
                        y,
                        get_pipeline,
                        CoxNet.predict,
                        splits=splits,
                        logger=logger,
                    )

                    c_index_outer = compute_metrics_on_splits(
                        result_outer,
                        [
                            partial2(c_index, X=X, y=y),
                        ],
                        y,
                    )[0]['c_index']
                except:
                    c_index_outer = None

            c_indexes_inner.append(c_index_inner)
            c_indexes_outer.append(c_index_outer)
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
                'c_indexes_inner': c_indexes_inner,
                'c_indexes_outer': c_indexes_outer,
                'hyperparameters': hyperparameters,
            }

        with open(get_hyperparameters_dependency_file(run_id), 'wb') as f:
            pickle.dump(hyperparameters_dependency, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', type=str)
    args = parser.parse_args()
    run(**vars(args))
