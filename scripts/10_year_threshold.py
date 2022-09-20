from hcve_lib.log_output import log_output, capture_output
from mlflow import set_tracking_uri, start_run, set_experiment, set_tag
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
from toolz import valmap

from config import GROUPS, MLFLOW_URL
from deps.common import get_data_cached
from hcve_lib.data import binarize_event
from hcve_lib.tracking import log_pickled
from hcve_lib.utils import binarize
from hcve_lib.utils import run_parallel
from notebooks.deps.binary_predictive_performance import get_endpoint_analysis_prediction


def main():
    data, metadata, X, y = get_data_cached()
    tte = 5 * 365
    y_bin = binarize_event(tte, y['data'])
    result = run_parallel(get_endpoint_analysis_prediction, valmap(lambda group_id: [group_id], GROUPS))
    run_parallel(
        get_cm_by_prediction,
        {name: [GROUPS[name], name, tte, y_bin, prediction]
         for name, prediction in result.items()},
    )


def get_cm_by_prediction(group_id, name, tte, y_bin, prediction):
    set_tracking_uri(MLFLOW_URL)
    set_experiment('threshold_analysis')
    with start_run(run_name=group_id) as mlflow:
        set_tag('tte_years', tte / 365)
        set_tag('method_name', name)
        with capture_output() as buffer:
            cm_by_threshold = run_parallel(
                get_cm_by_threshold,
                {threshold: [
                    y_bin,
                    threshold,
                    prediction,
                ]
                 for threshold in sorted(set(prediction['y_score']))},
            )
            log_pickled(cm_by_threshold, 'result')
        log_output(buffer())


def get_cm_by_threshold(y_bin, threshold, prediction):
    print('.')
    y_pred = binarize(prediction['y_score'], threshold)
    cm = confusion_matrix(y_bin, y_pred.loc[y_bin.index])
    print(f'{cm=}')
    return cm


def plot_(cm):
    _cm = [
        ['', 'PP', 'PN'],
        ['TP', *cm[0]],
        ['TN', *cm[1]],
    ]
    print(tabulate(_cm, tablefmt='fancy_grid'))
    print()


if __name__ == '__main__':
    main()
