import argparse
from typing import List

from mlflow import get_experiment_by_name, start_run, set_tracking_uri, set_tag

from common import log_result
from deps.common import get_variables
from hcve_lib.cv import train_test_filter
# noinspection PyUnresolvedReferences
from deps.ignore_warnings import *
from deps.prediction import run_prediction
from pipelines import get_pipelines


def run_lco(selected_methods: List[str]):
    set_tracking_uri('http://localhost:5000')
    experiment = get_experiment_by_name('reproduce')
    data, metadata, X, y = get_variables()
    cv = train_test_filter(
        data,
        train_filter=lambda _data: _data['STUDY'].isin(['HEALTHABC', 'PREDICTOR', 'PROSPER']),
        test_filter=lambda _data: _data['STUDY'] == 'ASCOT'
    )
    for method_name in selected_methods:
        with start_run(run_name=method_name, experiment_id=experiment.experiment_id):
            set_tag('method_name', method_name)
            current_method = get_pipelines()[method_name]

            result = run_prediction(
                cv,
                X,
                current_method.process_y(y),
                current_method.get_estimator,
                current_method.predict,
                n_jobs=1,
            )
            log_result(X, y, current_method, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('selected_methods', metavar='METHOD', type=str, nargs='+')
    args = parser.parse_args()
    run_lco(**vars(args))
