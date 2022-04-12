import logging
import pickle

from mlflow import set_tracking_uri, set_experiment

from deps.common import get_data_cached
# noinspection PyUnresolvedReferences
from hcve_lib.custom_types import Method
from deps.logger import logger
from deps.pipelines import XGBClassifierMethod, GB
from hcve_lib.cv import series_to_target, configuration_to_params
from hcve_lib.utils import loc, get_fraction_missing

set_tracking_uri('http://localhost:5000')
logger.setLevel(logging.INFO)
set_experiment('classify_cohort')

PICKLE_FILE = './data/cohort_models.data'
current_method = GB
method_name = 'xgb'
features_to_remove = ['CREA', 'HAF', 'SOK', 'CI', 'DRK', 'TRT_LIP', 'QRS']

data, metadata, X, y = get_data_cached()

X_grouped = X.groupby(data['STUDY'])

with open(PICKLE_FILE, 'rb') as f:
    models_per_cohort = pickle.load(f)

hyperparameters = {
    'PROSPER': {
        'estimator__inner': {
            'learning_rate': 0.4411298003819883,
            'max_depth': 2,
            'n_estimators': 112,
            'min_samples_split': 20,
            'min_samples_leaf': 49,
            'max_features': 'sqrt',
            'subsample': 0.5714910144705735
        }
    },
    'PREDICTOR': {
        'estimator__inner': {
            'learning_rate': 0.690747038405064,
            'max_depth': 6,
            'n_estimators': 119,
            'min_samples_split': 23,
            'min_samples_leaf': 94,
            'max_features': 'auto',
            'subsample': 0.4973416267980338
        }
    },
    'HVC': {
        'estimator__inner': {
            'learning_rate': 0.5929741272796457,
            'max_depth': 1,
            'n_estimators': 55,
            'min_samples_split': 9,
            'min_samples_leaf': 23,
            'max_features': 'log2',
            'subsample': 0.9079325525201548
        }
    },
    'HEALTHABC': {
        'estimator__inner': {
            'learning_rate': 0.10830341135304375,
            'max_depth': 10,
            'n_estimators': 125,
            'min_samples_split': 21,
            'min_samples_leaf': 32,
            'max_features': 'log2',
            'subsample': 0.2632207908276624
        }
    },
    'ASCOT': {
        'estimator__inner': {
            'learning_rate': 0.22770039915233234,
            'max_depth': 6,
            'n_estimators': 143,
            'min_samples_split': 21,
            'min_samples_leaf': 86,
            'max_features': 'log2',
            'subsample': 0.790282007307827
        }
    },
    'FLEMENGHO': {
        'estimator__inner': {
            'learning_rate': 0.5446561100266527,
            'max_depth': 1,
            'n_estimators': 102,
            'min_samples_split': 7,
            'min_samples_leaf': 162,
            'max_features': 'log2',
            'subsample': 0.9129044966995755
        }
    }
}


def train_model(X_cohort, hyperparameters):
    pipeline = current_method.get_estimator(X_cohort)
    X_cohort_missing_removed = X_cohort.copy()
    for column in X_cohort:
        if get_fraction_missing(X_cohort[column]) > 0.7:
            X_cohort_missing_removed.drop(columns=column, inplace=True)
    pipeline.set_params(**configuration_to_params(hyperparameters))
    pipeline.fit(X_cohort_missing_removed, loc(X_cohort_missing_removed.index, y)['data'])
    return {
        'pipeline': pipeline,
        'X_columns': X_cohort_missing_removed.columns,
    }


for cohort_name, X_cohort in X_grouped:
    models_per_cohort[cohort_name] = train_model(X_cohort, hyperparameters[cohort_name])

with open(PICKLE_FILE, 'wb') as f:
    pickle.dump(
        models_per_cohort,
        f,
    )
