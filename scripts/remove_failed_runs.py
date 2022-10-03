import argparse

from hcve_lib.tracking import get_subruns
from hcve_lib.utils import random_seed
from mlflow import set_tracking_uri, search_runs, delete_run

from deps.constants import RANDOM_STATE, TRACKING_URI

random_seed(RANDOM_STATE)
set_tracking_uri(TRACKING_URI)

parser = argparse.ArgumentParser()
parser.add_argument('--remove-unfinished', action='store_true')
args = parser.parse_args()

runs = search_runs(
    filter_string='attributes.status = "FAILED"',
    search_all_experiments=True,
    output_format='list',
)

if args.remove_unfinished:
    runs += search_runs(
        filter_string='attributes.status = "RUNNING"',
        search_all_experiments=True,
        output_format='list',
    )

for run in runs:
    subruns = get_subruns(run.info.run_id)

    for subrun in subruns.values():
        delete_run(subrun.info.run_id)
        print('.', end='')

    delete_run(run.info.run_id)
    print('.', end='')
