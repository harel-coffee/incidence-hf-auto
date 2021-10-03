import yaml
from pandas import read_csv, DataFrame

from hcve_lib.data import get_feature_subset, \
    sanitize_data_inplace, get_feature_names


def load_data() -> DataFrame:
    data = get_feature_subset(
        load_all_data(),
        get_feature_names(load_metadata()),
    )
    data = data[data['HTA'] == 0]
    return data


def load_all_data():
    data = read_csv('/home/sitnarf/Nextcloud/Cohorts/HOMAGE/homage_170314.csv', low_memory=False)
    sanitize_data_inplace(data)
    return data


def load_metadata():
    with open("./metadata.yaml", 'r') as stream:
        metadata = yaml.safe_load(stream)
    return metadata
