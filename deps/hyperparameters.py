import json
from functools import cache
from typing import Dict

HYPERPARAMETERS_FILE = './data/hyperparameters.json'


@cache
def get_all_hyperparameters() -> Dict:
    with open(HYPERPARAMETERS_FILE, 'r') as f:
        return json.load(f)


def get_hyperparameters(
    method_name: str,
    splits: str,
    split_name: str = None,
) -> Dict:
    base = get_all_hyperparameters()[splits][method_name]

    if split_name:
        return base[split_name]
    else:
        return base
