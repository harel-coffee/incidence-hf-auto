import json

HYPERPARAMETERS_FILE = './data/hyperparameters.json'


def get_hyperparameters():
    with open(HYPERPARAMETERS_FILE, 'r') as f:
        return json.load(f)
