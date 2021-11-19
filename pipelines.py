from typing import Dict, Tuple, Optional, List, Type

import numpy
import torchtuples as tt
from optuna import Trial
from pandas import DataFrame
from pycox.models import CoxPH
from sklearn.base import BaseEstimator
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from torch import nn

from deps.constants import RANDOM_STATE, CACHE_DIR
from deps.custom_types import MethodDefinition
from dsm import DeepCoxMixtures
from hcve_lib.custom_types import Estimator
from hcve_lib.cv import predict_survival, CROSS_VALIDATE_KEY, predict_survival_dsm
from hcve_lib.data import to_survival_y_records
from hcve_lib.pipelines import TransformTarget
from hcve_lib.transformers import MiceForest
from hcve_lib.utils import remove_column_prefix
from hcve_lib.wrapped_sklearn import DFCoxnetSurvivalAnalysis, DFPipeline, DFColumnTransformer, DFSimpleImputer, \
    DFOrdinalEncoder, DFStacking, DFWrapped


def get_pipelines() -> Dict[str, Type[MethodDefinition]]:
    return {
        'coxnet': CoxNet,
        'gb': GB,
        'rsf': RSF,
        'stacking': Stacking,
        'gb_limited': GBLimited,
        'dcm': DeepCoxMixtureMethod,
        # 'pycox': {
        #     'get_estimator': lambda X, verbose=0: get_pycox_pipeline(X, verbose=verbose),
        #     'process_y': to_survival_y_pair,
        #     'predict': predict_nn,
        # }
    }


class Stacking(MethodDefinition):

    @staticmethod
    def get_estimator(X: DataFrame, verbose=0, advanced_impute=False):
        return get_standard_pipeline(
            TransformTarget(
                DFStacking(
                    DecisionTreeRegressor(), [
                        ('coxnet', DFCoxnetSurvivalAnalysis),
                        ('gb', GradientBoostingSurvivalAnalysis)
                    ]
                ), to_survival_y_records
            ),
            X,
            advanced_impute=advanced_impute,
        )

    @staticmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]:
        hyperparameters = {
            CROSS_VALIDATE_KEY: {
                'missing_fraction': trial.suggest_uniform(
                    f'{CROSS_VALIDATE_KEY}__missing_fraction', 0.1, 1
                ),
            },
            'estimator__inner': {
                'l1_ratio': 1 - trial.suggest_loguniform('estimator_n_alphas', 0.1, 1),
            }
        }
        return trial, hyperparameters

    process_y = to_survival_y_records
    predict = predict_survival


class CoxNet(MethodDefinition):

    @staticmethod
    def get_estimator(X: DataFrame, verbose=True, advanced_impute=False):
        return get_standard_pipeline(
            TransformTarget(
                DFCoxnetSurvivalAnalysis(fit_baseline_model=True, verbose=verbose),
                to_survival_y_records
            ),
            X,
            advanced_impute=advanced_impute,
        )

    @staticmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]:
        hyperparameters = {
            CROSS_VALIDATE_KEY: {
                'missing_fraction': trial.suggest_uniform(
                    f'{CROSS_VALIDATE_KEY}__missing_fraction', 0.1, 1
                ),
            },
            'estimator__inner': {
                'l1_ratio': 1 - trial.suggest_loguniform('estimator_l1_ratio', 0.1, 1),
                'alphas': [trial.suggest_loguniform('estimator_alphas', 10**-1, 10**3)]
            }
        }
        return trial, hyperparameters

    process_y = to_survival_y_records
    predict = predict_survival


class GB(MethodDefinition):

    @staticmethod
    def get_estimator(X: DataFrame, verbose=0, advanced_impute=False):
        return get_standard_pipeline(
            TransformTarget(
                GradientBoostingSurvivalAnalysis(
                    random_state=RANDOM_STATE,
                    verbose=verbose,
                ),
                to_survival_y_records,
            ),
            X,
        )

    @staticmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]:
        hyperparameters = {
            CROSS_VALIDATE_KEY: {
                'missing_fraction': trial.suggest_uniform(
                    f'{CROSS_VALIDATE_KEY}_missing_fraction', 0.1, 1
                ),
            },
            'estimator__inner': {
                'learning_rate': trial.suggest_uniform('estimator_learning_rate', 0, 1),
                'max_depth': trial.suggest_int('estimator_max_depth', 1, 10),
                'n_estimators': trial.suggest_int('estimator_n_estimators', 5, 200),
                'min_samples_split': trial.suggest_int('estimator_min_samples_split', 2, 30),
                'min_samples_leaf': trial.suggest_int('estimator_min_samples_leaf', 1, 200),
                'max_features': trial.suggest_categorical(
                    'estimator_max_features', ['auto', 'sqrt', 'log2']
                ),
                'subsample': trial.suggest_uniform('estimator_subsample', 0.1, 1),
            }
        }
        return trial, hyperparameters

    process_y = to_survival_y_records
    predict = predict_survival


class GBLimited(GB):

    @staticmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]:
        hyperparameters = {
            CROSS_VALIDATE_KEY: {
                'missing_fraction': trial.suggest_uniform(
                    f'{CROSS_VALIDATE_KEY}_missing_fraction', 0.1, 1
                ),
            },
            'estimator__inner': {
                'learning_rate': trial.suggest_uniform('estimator_learning_rate', 0, 1),
                'min_samples_leaf': trial.suggest_int('estimator_min_samples_leaf', 1, 200),
                'max_features': trial.suggest_categorical(
                    'estimator_max_features', ['auto', 'sqrt', 'log2']
                ),
                'subsample': trial.suggest_uniform('estimator_subsample', 0.1, 1),
            }
        }
        return trial, hyperparameters


class RSF(MethodDefinition):

    @staticmethod
    def get_estimator(X: DataFrame, verbose=0, advanced_impute=False):
        return get_standard_pipeline(
            RandomSurvivalForest(random_state=RANDOM_STATE),
            X,
        )

    @staticmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]:
        hyperparameters = {
            'estimator': {
                'n_estimators': trial.suggest_int('estimator_n_estimators', 5, 200),
                'max_depth': trial.suggest_int('estimator_max_depth', 1, 30),
                'min_samples_split': trial.suggest_int('estimator_min_samples_split', 2, 30),
                'min_samples_leaf': trial.suggest_int('estimator_min_samples_leaf', 1, 200),
                'max_features': trial.suggest_categorical(
                    'estimator_max_features', ['auto', 'sqrt', 'log2']
                ),
                'oob_score': trial.suggest_categorical('estimator_oob_score', [True, False]),
            }
        }
        return trial, hyperparameters

    process_y = to_survival_y_records
    predict = predict_survival


def get_standard_pipeline(
    estimator: Estimator,
    X: DataFrame,
    advanced_impute: bool = False,
) -> DFPipeline:
    return DFPipeline(
        [
            *(get_mice_imputer(X) if advanced_impute else get_basic_imputer(X)),
            *get_encoder(X),
            ('estimator', estimator),
        ],
        # memory=CACHE_DIR,
    )


def get_preprocessing(X: DataFrame) -> Tuple:
    categorical_features, continuous_features = categorize_features(X)

    pipeline = [
        *get_basic_imputer(X),
        *get_encoder(X),
    ]

    return pipeline, categorical_features, continuous_features


def get_basic_imputer(X: DataFrame) -> List[Tuple]:
    categorical_features, continuous_features = categorize_features(X)
    return [
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
            'imputer_remove_prefix',
            FunctionTransformer(remove_column_prefix),
        ),
    ]


def get_mice_imputer(_) -> List[Tuple]:
    return [('imputer', MiceForest(random_state=5465132, iterations=5))]


def get_encoder(X: DataFrame) -> List[Tuple]:
    categorical_features, continuous_features = categorize_features(X)
    return [
        (
            'encoder',
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
            'encoder_remove_prefix',
            FunctionTransformer(remove_column_prefix),
        ),
    ]


def categorize_features(X: DataFrame) -> Tuple[List[str], List[str]]:
    categorical_features = [
        column_name for column_name in X.columns
        if X[column_name].dtype.name == 'object' or X[column_name].dtype.name == 'category'
    ]
    continuous_features = [
        column_name for column_name in X.columns if column_name not in categorical_features
    ]
    return categorical_features, continuous_features


def transform_X(input):
    return numpy.array(input).astype('float32')


class NNWrapper(CoxPH):
    net: Optional[nn.Module]

    def __init__(self, net=None, *args, **kwargs):
        self.net = net
        self.args = args
        self.kwargs = kwargs

    def fit(self, X, y, **kwargs):
        if not self.net:
            self.net = make_net(X.shape[1])

        super().__init__(self.net, tt.optim.Adam, *self.args, **self.kwargs)
        super().fit(X.astype('float32'), y)
        return self


def make_net(in_features: int) -> nn.Module:
    num_nodes = [32, 32]
    out_features = 1
    batch_norm = True
    dropout = 0.1
    return tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)


def get_pycox_pipeline(X: DataFrame, verbose: int = 0):
    preprocessing, categorical_features, continuous_features = get_preprocessing(X)

    return DFPipeline(
        [
            *preprocessing,
            ('standard', StandardScaler()),
            ('to_numpy', FunctionTransformer(transform_X)),
            ('estimator', NNWrapper()),
        ],
        # memory=CACHE_DIR
    )


class DeepCoxMixtureAdapter(DFWrapped, BaseEstimator, DeepCoxMixtures):
    model = None

    def fit(self, X, y, **kwargs):
        self.fitted_feature_names = self.get_fitted_feature_names(X)
        t = y['data']['tte'].to_numpy().astype('float32')
        e = y['data']['label'].to_numpy()
        super(DFWrapped, self).fit(X.to_numpy(), t, e, iters=50)
        return self

    def predict(self, X):
        return 1 - super().predict_survival(X.to_numpy(), 1 * 365)

    def predict_survival(self, X, t):
        super().predict_survival(X.to_numpy(), t)

    def predict_survival_function(self, X):
        for x in X:
            yield lambda t: self.predict_survival(x, t)


class DeepCoxMixtureMethod(MethodDefinition):

    @staticmethod
    def get_estimator(X: DataFrame, verbose=True, advanced_impute=False):
        return get_standard_pipeline(
            DeepCoxMixtureAdapter(),
            X,
            advanced_impute=advanced_impute,
        )

    @staticmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]:
        hyperparameters = {
            CROSS_VALIDATE_KEY: {
                'missing_fraction': trial.suggest_uniform(
                    f'{CROSS_VALIDATE_KEY}__missing_fraction', 0.1, 1
                ),
            },
            'estimator': {
                'k': trial.suggest_int('estimator_k', 1, 5),
                'layers': [trial.suggest_int('layers_k', 3, 100)],
            }
        }
        return trial, hyperparameters

    process_y = to_survival_y_records
    predict = predict_survival_dsm
