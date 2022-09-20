from typing import List, Union, Tuple

import pandas
import yaml
from pandas import read_csv, DataFrame

from config import DATA_MAIN_PATH, DATA_FLEMENGHO_PATH
from deps.memory import memory
from deps.logger import logger
from hcve_lib.data import get_feature_subset, \
    sanitize_data_inplace, get_variable_identifier, Metadata, get_X
from hcve_lib.formatting import format_number
from hcve_lib.functional import t, statements
from hcve_lib.preprocessing import Step
from hcve_lib.preprocessing import perform, log_step, remove_cohorts


def get_homage_X(
    data: DataFrame,
    metadata: Metadata,
) -> DataFrame:
    X = get_X(data, metadata)
    return X.drop(['IDNR', 'VISIT', 'STUDY_NUM', 'STUDY'], axis=1)


def load_data(metadata: Metadata) -> DataFrame:
    data = perform(
        [
            Step(
                action=lambda _: load_all_data(),
                log=log_step('Raw data', metadata),
            ),
            Step(
                action=lambda current: current[current['VISIT'] == 'BASELINE'],
                log=log_step('Baseline visit kept', metadata),
            ),
            Step(
                action=lambda current:
                statements(to_drop := current[current['STUDY'] == 'HULL_LIFELAB'], current.drop(to_drop.index)),
                log=log_step('Dropping HULL LIFE LAB cohort', metadata),
            ),
            Step(
                action=lambda current: current[(current['AGE'] >= 30) & (current['AGE'] <= 80)],
                log=log_step('Only 30-80 age', metadata),
            ),
            Step(
                action=lambda current: remove_cohorts(current, ['leitzaran', 'hfgr', 'timechf'], metadata),
                log=log_step('HF cohorts removed', metadata),
            ),
            Step(
                action=lambda current: current[(current['HHF'] == 0) | (current['STUDY_NUM'] == 18)],
                log=log_step('HF individuals at baseline removed', metadata),
            ),
            Step(
                action=lambda current:
                remove_cohorts(current, ['epath', 'iblomaved', 'stophf', 'dyda', 'biomarcoeurs'], metadata),
                log=log_step('No outcome cohorts removed', metadata),
            ),
            Step(
                action=lambda current: current[
                    # (~current['FCV'].isna() & ~current['FUFCV'].isna()) &
                    (~current['NFHF'].isna() & ~current['FUNFHF'].isna())],
                log=log_step('Missing outcome individuals removed', metadata),
            ),
            Step(
                action=lambda current: remove_cohorts(
                    current, ['adelhyde', 'gecoh', 'r2c2', 'reve(1-2)', 'stanislas', 'styrianvitd'], metadata
                ),
                log=log_step('Missing HF data cohorts removed', metadata),
            ),
            Step(
                action=lambda current: current[~current['SBP'].isna() | ~current['DBP'].isna()],
                log=log_step('Missing blood pressure measurements', metadata),
            ),
            Step(
                action=lambda current: fill_missing_pp(current),
                log=lambda logger_,
                current,
                previous: logger_.info(
                    f'Providing missing PP for '
                    f'{len(previous[previous["PP"].isna()]) - len(current[current["PP"].isna()])}'
                    f'individuals\n'
                ),
            ),
            Step(
                action=lambda current: current[current['FUNFHF'] > 0],
                log=log_step('Removes individuals with 0 follow up time', metadata),
            ),
            Step(
                log=lambda logger,
                current,
                _: logger.info(
                    f'Final dataset\n'
                    f'\tn individuals={format_number(len(current))}\n'
                    f'\tn cohorts={len(current["STUDY_NUM"].unique())}\n'
                ),
            ),
            Step(
                action=auto_convert_category,
                log=lambda logger,
                current,
                past: logger.info(
                    'Convert before:\n' + str(past.dtypes.map(str).value_counts()) + "Now:\n " +
                    str(current.dtypes.map(str).value_counts())
                )
            )
        ],
        logger=logger
    )
    missing_or_irrelevant = [
        'PACKYEARS',
        'NYHA',
        'HAP',
        'HMI',
        'HIHD',
        'HCABG',
        'HPTCA',
        'HHF',
        'HSTROKE',
        'HTIA',
        'HVALVE',
        'HCV_OTHER',
        'HYPERCHOL',
        'DEPRESSION',
        'TRT_STAT',
        'TRT_FIB',
        'TRT_ANTIPLT',
        'TRT_ASPIRIN',
        'TRT_AC',
        'HTA',
        'LVH',
        'HB',
        'HBA1C',
        'HCT',
        'WBC',
        'RBC',
        'PLT',
        'CRP',
        'NA',
        'K',
        'AST',
        'ALT',
        'BNP',
        'NTPROBNP',
        'ALBU',
        'BICARB',
        'TOT_PROT',
        'LPA',
        'FIBRINOGEN',
        'HOMOCYST',
        'EGFR'
    ]

    data.drop(['DBIRTH', 'DVISIT', *missing_or_irrelevant], axis=1, inplace=True)

    return data


def auto_convert_category(data: DataFrame) -> DataFrame:
    data_new = data.copy()
    for column in data_new.columns:
        if len(data_new[column].unique()) < 10:
            data_new.loc[:, column] = data_new[column].astype('category')
        else:
            try:
                data_new.loc[:, column] = data_new[column].astype('float')
            except TypeError:
                pass
    return data_new


load_data_cached = memory.cache(load_data)


def fill_missing_pp(current: DataFrame) -> DataFrame:
    new_data_frame = current.copy()
    missing_pp = current['PP'].isna()
    new_data_frame.loc[missing_pp, 'PP'] = current[missing_pp]['SBP'] - current[missing_pp]['DBP']
    return new_data_frame


def load_all_data():
    data = get_feature_subset(
        load_raw_data(),
        get_variable_identifier(load_metadata()),
    )
    print(data)
    return data


def load_raw_data():
    data_main = read_csv_(DATA_MAIN_PATH)
    sanitize_data_inplace(data_main)
    data_main.drop(data_main[data_main['STUDY'] == 'FLEMENGHO'].index, inplace=True)

    data_flemengho = load_flemengho()
    return pandas\
        .concat([data_main, data_flemengho])\
        .set_index('IDNR', drop=False)


def load_flemengho():
    data_flemengho = read_csv_(DATA_FLEMENGHO_PATH)
    sanitize_data_inplace(data_flemengho)
    data_flemengho.rename({'PR': 'HR'}, inplace=True, axis=1)
    return data_flemengho


def read_csv_(path: str) -> DataFrame:
    return read_csv(path, low_memory=False, parse_dates=['dvisit', 'dbirth'])


def load_metadata():
    with open("./metadata.yaml", 'r') as stream:
        metadata = yaml.safe_load(stream)
    return metadata


def group_by_study(data: DataFrame, X: DataFrame = None):
    if X is None:
        X = data

    return X.groupby(data['STUDY'])


def get_30_to_80(_X):
    return _X[(_X['AGE'] >= 30) & (_X['AGE'] <= 80)]
