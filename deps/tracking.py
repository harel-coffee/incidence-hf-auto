from mlflow import start_run, log_metrics

from hcve_lib.tracking import encode_run_name


def log_splits_as_subruns(metrics_folds, experiment_id=None):
    for fold_name, fold_metrics in metrics_folds.items():
        with start_run(
            run_name=encode_run_name(fold_name), nested=True, experiment_id=experiment_id
        ):
            log_metrics(fold_metrics)
