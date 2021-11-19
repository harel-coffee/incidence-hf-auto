from deps.data import load_metadata, get_homage_X, load_data
from deps.memory import memory
from hcve_lib.data import get_survival_y


def get_variables():
    metadata = load_metadata()
    data = load_data(metadata)
    X, y = (
        get_homage_X(data, metadata),
        get_survival_y(data, 'NFHF', metadata),
    )
    return data, metadata, X, y


get_variables_cached = memory.cache(get_variables)
