from hcve_lib.data import to_survival_y_records, get_survival_y

from deps.data import load_metadata, load_data_cached, get_homage_X
from deps.memory import memory


def get_variables():
    metadata = load_metadata()
    data = load_data_cached(metadata)
    X, y = (
        get_homage_X(data, metadata),
        get_survival_y(data, 'NFHF', metadata),
    )
    return data, metadata, X, y


get_variables_cached = memory.cache(get_variables)

RANDOM_STATE = 502141521
CACHE_DIR = '.project-cache'
