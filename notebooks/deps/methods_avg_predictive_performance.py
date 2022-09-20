from deps.constants import RANDOM_STATE
from hcve_lib.evaluation_functions import compute_metric_result
from hcve_lib.metrics import BootstrappedMetric
from hcve_lib.metrics import CIndex
from hcve_lib.tracking import load_run_results


def get_method_comparison(run_id, X, y, iterations=10):
    return compute_metric_result(
        BootstrappedMetric(
            CIndex(),
            RANDOM_STATE,
            iterations=iterations,
            return_summary=False,
        ),
        y,
        load_run_results(run_id),
    )
