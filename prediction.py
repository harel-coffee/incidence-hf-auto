# %%
import pandas
import pickle
from matplotlib import pyplot
from sklearn.preprocessing import FunctionTransformer
from sksurv.ensemble import RandomSurvivalForest

from deps.data import load_metadata, load_data_cached
from deps.logger import logger
from hcve_lib.cv import cross_validate, predict_survival, kfold_cv, filter_missing_features
from hcve_lib.data import to_survival_y_records, get_X, get_survival_y, format_identifier_long
from hcve_lib.evaluation_functions import compute_metrics, c_index
from hcve_lib.functional import pipe, statements
from hcve_lib.utils import remove_column_prefix
from hcve_lib.wrapped_sklearn import DFColumnTransformer, DFPipeline, DFSimpleImputer, DFOrdinalEncoder, \
    DFVarianceThreshold
# noinspection PyUnresolvedReferences
from ignore_warnings import *

pandas.set_option("display.max_columns", None)
pyplot.rcParams['figure.facecolor'] = 'white'

logger.setLevel('DEBUG')

metadata = load_metadata()
data = load_data_cached(metadata)

X, y = pipe(
    (
        get_X(data, metadata),
        to_survival_y_records(get_survival_y(data, 'NFHF', metadata)),
    ),
)

for column in X.columns:
    if len(X[column].unique()) < 10:
        X.loc[:, column] = X[column].astype('category')
    print(f'{X[column].dtype}: {format_identifier_long(column, metadata)}')

categorical_features = [
    column_name for column_name in X.columns if X[column_name].dtype == 'category'
]

continuous_features = [
    column_name for column_name in X.columns if column_name not in categorical_features
]

pipeline = DFPipeline(
    [
        ('zero_variance', DFVarianceThreshold()),
        (
            'imputer',
            DFColumnTransformer(
                [
                    (
                        'categorical',
                        DFSimpleImputer(strategy='most_frequent'),
                        categorical_features,
                    ),
                    (
                        'continuous',
                        DFSimpleImputer(strategy='mean'),
                        continuous_features,
                    ),
                ],
            )
        ),
        (
            'remove_prefix',
            FunctionTransformer(remove_column_prefix),
        ),
        # ('imputer', MiceForest(random_state=5465132, iterations=5)),
        (
            'scaler',
            DFColumnTransformer(
                [(
                    'categorical',
                    DFOrdinalEncoder(),
                    categorical_features,
                )],
                remainder='passthrough',
            )
        ),
        (
            'remove_prefix2',
            FunctionTransformer(remove_column_prefix),
        ),
        ('estimator', RandomSurvivalForest(verbose=5)),
    ],
    # memory='.project-cache'
)

X, y = pipe(
    (
        get_X(data, metadata),
        to_survival_y_records(get_survival_y(data, 'NFHF', metadata)),
    ),
)

# cv = train_test(
#     data,
#     train_filter=lambda _data: _data['STUDY'].isin(['HEALTHABC', 'PREDICTOR', 'PROSPER']),
#     test_filter=lambda _data: _data['STUDY'] == 'ASCOT'
# )

# cv = lco_cv(data.groupby('STUDY'))
cv = kfold_cv(data, n_splits=10, shuffle=True, random_state=243)
# cv = lm_cv(data.groupby('STUDY'))

folds_prediction = cross_validate(
    X,
    y,
    pipeline,
    predict_survival,
    cv,
    lambda x_train, x_test: statements(mask := filter_missing_features(x_train, x_test), ),
    # n_jobs=1,
)

with open('./data/prediction.data', 'wb') as file:
    pickle.dump(compute_metrics(folds_prediction, [c_index]), file)
