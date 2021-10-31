from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest

from deps.methods import coxnet_optuna, gb_optuna, rsf_optuna
from hcve_lib.data import to_survival_y_records, get_survival_y

from deps.data import load_metadata, load_data_cached, get_homage_X
from deps.memory import memory
from hcve_lib.wrapped_sklearn import DFCoxnetSurvivalAnalysis


def get_variables():
    metadata = load_metadata()
    data = load_data_cached(metadata)
    X, y = (
        get_homage_X(data, metadata),
        to_survival_y_records(get_survival_y(data, 'NFHF', metadata)),
    )
    return data, metadata, X, y


get_variables_cached = memory.cache(get_variables)
METHODS_DEFINITIONS = {
    'coxnet': {
        'get_estimator': DFCoxnetSurvivalAnalysis,
        'optuna': coxnet_optuna,
    },
    'gb': {
        'get_estimator': lambda: GradientBoostingSurvivalAnalysis(random_state=RANDOM_STATE),
        'optuna': gb_optuna,
    },
    'rsf': {
        'get_estimator': lambda: RandomSurvivalForest(random_state=RANDOM_STATE),
        'optuna': rsf_optuna,
    },
    'nn': {
        'get_estimator':...
    }
}

RANDOM_STATE = 502141521
