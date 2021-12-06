from typing import List, Iterable, Tuple, Sequence

from deps.data import load_metadata, get_homage_X, load_data
from deps.logger import logger
from deps.memory import memory
from hcve_lib.data import get_survival_y


def get_variables(remove_cohorts: Sequence[str] = tuple()):
    metadata = load_metadata()
    data = load_data(metadata)

    if len(remove_cohorts) > 0:
        logger.info(f'Manually dropping cohorts: {", ".join(remove_cohorts)}')
        data_selected = data[~data['STUDY'].isin(remove_cohorts)]
        data_selected['STUDY'] = data_selected['STUDY'].cat.remove_unused_categories()
    else:
        data_selected = data

    X, y = (
        get_homage_X(data_selected, metadata),
        get_survival_y(data_selected, 'NFHF', metadata),
    )
    return data_selected, metadata, X, y


get_variables_cached = memory.cache(get_variables)
