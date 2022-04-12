import argparse
from typing import List

from mlflow import get_experiment_by_name, start_run, set_tracking_uri, set_tag

from common import log_result
from deps.common import get_data_cached
from deps.data import get_homage_X
from hcve_lib.splitting import kfold_stratified_cv
# noinspection PyUnresolvedReferences
from deps.ignore_warnings import *
from deps.prediction import run_prediction
from hcve_lib.data import get_survival_y
from deps.pipelines import get_pipelines


def run(selected_methods: List[str]):
    set_tracking_uri('http://localhost:5000')
    experiment = get_experiment_by_name('10_fold')
    data, metadata, X, y = get_data_cached()
    for method_name in selected_methods:
        with start_run(run_name=method_name, experiment_id=experiment.experiment_id):
            set_tag('method_name', method_name)
            for study_name, study_data in data.groupby('STUDY'):
                with start_run(run_name=study_name, experiment_id=experiment.experiment_id, nested=True):
                    X_, y_ = (
                        get_homage_X(study_data, metadata),
                        get_survival_y(study_data, 'NFHF', metadata),
                    )
                    cv = kfold_stratified_cv(study_data, y.loc[study_data.index]['label'], n_splits=10)
                    for train_index, test_index in cv.values():
                        print(study_data.iloc[test_index]['NFHF'].value_counts())

                    current_method = get_pipelines()[method_name]
                    result = run_prediction(
                        cv,
                        X_,
                        current_method['process_y'](y_),
                        current_method['get_estimator'],
                        current_method['predict'],
                        n_jobs=-1,
                    )
                    log_result(X, y, current_method, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('selected_methods', metavar='METHOD', type=str, nargs='+')
    args = parser.parse_args()
    run(**vars(args))
