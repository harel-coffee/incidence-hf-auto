from pandas import DataFrame
from sklearn.preprocessing import FunctionTransformer

from hcve_lib.custom_types import Estimator
from hcve_lib.utils import remove_column_prefix
from hcve_lib.wrapped_sklearn import DFColumnTransformer, DFPipeline, DFSimpleImputer, DFOrdinalEncoder
# noinspection PyUnresolvedReferences
from ignore_warnings import *


def get_pipeline(estimator: Estimator, X: DataFrame) -> DFPipeline:
    categorical_features = [
        column_name for column_name in X.columns if X[column_name].dtype == 'category'
    ]

    continuous_features = [
        column_name for column_name in X.columns if column_name not in categorical_features
    ]
    return DFPipeline(
        [
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
                    [
                        (
                            'categorical',
                            DFOrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
                            categorical_features,
                        )
                    ],
                    remainder='passthrough',
                )
            ),
            (
                'remove_prefix2',
                FunctionTransformer(remove_column_prefix),
            ),
            ('estimator', estimator),
        ],
        # memory='.project-cache'
    )
