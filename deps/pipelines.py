from abc import ABC
from math import log, exp
from typing import Dict, Tuple, List, Type, Callable

import numpy
import numpy as np
import torch
from optuna import Trial
from pandas import DataFrame, Series, Index
from sklearn.base import BaseEstimator
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from torch import autograd
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from deps.constants import RANDOM_STATE
from deps.logger import logger
from dsm import DeepCoxMixtures, DeepSurvivalMachines
from dsm_original import DeepCoxMixtures as DeepCoxMixturesOriginal
from hcve_lib.custom_types import Estimator, Splitter, Method, SplitPrediction, Target, SplitInput
from hcve_lib.cv import predict_survival, CROSS_VALIDATE_KEY, predict_survival_dsm, predict_proba, predict_predict
from hcve_lib.data import to_survival_y_records, categorize_features
from hcve_lib.functional import reject_none_values, pipe
from hcve_lib.pipelines import TransformTarget, TransformerTarget
from hcve_lib.transformers import MiceForest
from hcve_lib.utils import remove_column_prefix, X_to_pytorch, loc
from hcve_lib.wrapped_sklearn import DFCoxnetSurvivalAnalysis, DFPipeline, DFColumnTransformer, DFSimpleImputer, \
    DFOrdinalEncoder, DFStacking, DFWrapped, DFStandardScaler
from networks import DeepSurv
from networks import NegativeLogLikelihood
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from utils import adjust_learning_rate, c_index
# noinspection PyUnresolvedReferences
import torch.optim as optim


def get_pipelines() -> Dict[str, Type[Method]]:
    return {
        'coxnet': CoxNet,
        'gb': GB,
        'rsf': RSF,
        'stacking': Stacking,
        'gb_limited': GBLimited,
        'dcm': DeepCoxMixtureMethod,
        'dcm_original': DeepCoxMixtureOriginalMethod,
        'dsm': DeepSurvivalMachinesMethod,
        'deep_surv': DeepSurvMethod,
        'pc': PooledCohortMethod
        # 'pycox': {
        #     'get_estimator': lambda X, verbose=0: get_pycox_pipeline(X, verbose=verbose),
        #     'process_y': to_survival_y_pair,
        #     'predict': predict_nn,
        # }
    }


class PooledCohortMethod(Method):

    @staticmethod
    def get_estimator(X: DataFrame, verbose=True):
        return DFPipeline([
            *get_basic_imputer(X),
            ('estimator', PooledCohort()),
        ])

    @staticmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]:
        raise Exception('Not possible to optimize')

    @staticmethod
    def predict(
        X: DataFrame,
        y: Target,
        split: SplitInput,
        model: Estimator,
    ) -> SplitPrediction:
        return SplitPrediction(
            split=split,
            X_columns=X.columns.tolist(),
            y_column=y['name'],
            y_score=model.predict(loc(split[1], X)),
            model=model,
        )


class PooledCohort(BaseEstimator):

    def fit(self, *args, **kwargs):
        pass

    def predict(self, X):
        X_ = X.copy()

        def predict_(row: Series) -> float:
            if row['AGE'] < 30:
                return np.nan

            bsug_adj = row['GLU'] * 18.018
            row['CHOL'] = row['CHOL'] * 38.67
            row['HDL'] = row['HDL'] * 38.67
            bmi = row['BMI']
            sex = row['SEX']
            qrs = row['QRS']
            trt_ht = row['TRT_AH']
            if row['SMK'] == 0 or row['SMK'] == 2:
                csmk = 0
            elif row['SMK'] == 1:
                csmk = 1
            ln_age = log(row['AGE'])
            ln_age_sq = ln_age**2
            ln_tchol = log(row['CHOL'])
            ln_hchol = log(row['HDL'])
            ln_sbp = log(row['SBP'])
            hdm = row['DIABETES']
            ln_agesbp = ln_age * ln_sbp
            ln_agecsmk = ln_age * csmk
            ln_bsug = log(bsug_adj)
            ln_bmi = log(bmi)
            ln_agebmi = ln_age * ln_bmi
            ln_qrs = log(qrs)

            if (sex == 1) and (trt_ht == 0): coeff_sbp = 0.91
            if (sex == 1) and (trt_ht == 1): coeff_sbp = 1.03
            if (sex == 2) and (trt_ht == 0): coeff_sbp = 11.86
            if (sex == 2) and (trt_ht == 1): coeff_sbp = 12.95
            if (sex == 2) and (trt_ht == 0): coeff_agesbp = -2.73
            if (sex == 2) and (trt_ht == 1): coeff_agesbp = -2.96

            if (sex == 1) and (hdm == 0): coeff_bsug = 0.78
            if (sex == 1) and (hdm == 1): coeff_bsug = 0.90
            if (sex == 2) and (hdm == 0): coeff_bsug = 0.91
            if (sex == 2) and (hdm == 1): coeff_bsug = 1.04

            if sex == 1:
                IndSum = ln_age * 41.94 + ln_age_sq * (
                    -0.88
                ) + ln_sbp * coeff_sbp + csmk * 0.74 + ln_bsug * coeff_bsug + ln_tchol * 0.49 + ln_hchol * (
                    -0.44
                ) + ln_bmi * 37.2 + ln_agebmi * (-8.83) + ln_qrs * 0.63

                HF_risk = 100 * (1 - (0.98752**exp(IndSum - 171.5)))
            elif sex == 2:
                IndSum = ln_age * 20.55 + ln_sbp * coeff_sbp + ln_agesbp * coeff_agesbp + csmk * 11.02 + ln_agecsmk * (
                    -2.50
                ) + ln_bsug * coeff_bsug + ln_hchol * (-0.07) + ln_bmi * 1.33 + ln_qrs * 1.06
                HF_risk = 100 * (1 - (0.99348**exp(IndSum - 99.73)))
            return HF_risk

        return X.apply(predict_, axis=1).dropna()


class Stacking(Method):

    @staticmethod
    def get_estimator(X: DataFrame, verbose=0, advanced_impute=False):
        return get_standard_steps(
            TransformTarget(
                DFStacking(
                    DecisionTreeRegressor(),
                    [('coxnet', DFCoxnetSurvivalAnalysis), ('gb', GradientBoostingSurvivalAnalysis)]
                ),
                to_survival_y_records,
            ),
            X,
            advanced_impute=advanced_impute,
        )

    @staticmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]:
        hyperparameters = {
            CROSS_VALIDATE_KEY: {
                'missing_fraction': trial.suggest_uniform(f'{CROSS_VALIDATE_KEY}__missing_fraction', 0.1, 1),
            },
            'estimator__inner': {
                'l1_ratio': 1 - trial.suggest_loguniform('estimator_n_alphas', 0.1, 1),
            }
        }
        return trial, hyperparameters

    process_y = to_survival_y_records
    predict = predict_survival


class CoxNet(Method):

    @staticmethod
    def get_estimator(X: DataFrame, random_state: int, verbose=True, advanced_impute=False):
        return DFPipeline(
            get_standard_steps(
                TransformTarget(
                    DFCoxnetSurvivalAnalysis(
                        fit_baseline_model=True,
                        verbose=verbose,
                        n_alphas=1,
                    ),
                    to_survival_y_records
                ),
                X,
                advanced_impute=advanced_impute,
            )
        )

    @staticmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]:
        hyperparameters = {
            CROSS_VALIDATE_KEY: {
                'missing_fraction': trial.suggest_uniform(f'{CROSS_VALIDATE_KEY}__missing_fraction', 0.1, 1),
            },
            'estimator__inner': {
                'l1_ratio': 1 - trial.suggest_loguniform('estimator_l1_ratio', 0.1, 1),
                'alphas': [trial.suggest_loguniform('estimator_alphas', 10**-1, 10**3)]
            }
        }
        return trial, hyperparameters

    process_y = to_survival_y_records
    predict = predict_survival


class GB(Method):

    @staticmethod
    def get_estimator(X: DataFrame, verbose=0, advanced_impute=False):
        return DFPipeline(
            get_standard_steps(
                TransformTarget(
                    GradientBoostingSurvivalAnalysis(
                        random_state=RANDOM_STATE,
                        verbose=verbose,
                    ),
                    to_survival_y_records,
                ),
                X,
            )
        )

    @staticmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]:
        hyperparameters = {
            CROSS_VALIDATE_KEY: {
                'missing_fraction': trial.suggest_uniform(f'{CROSS_VALIDATE_KEY}_missing_fraction', 0.1, 1),
            },
            'estimator__inner': {
                'learning_rate': trial.suggest_uniform('estimator_learning_rate', 0, 1),
                'max_depth': trial.suggest_int('estimator_max_depth', 1, 10),
                'n_estimators': trial.suggest_int('estimator_n_estimators', 5, 200),
                'min_samples_split': trial.suggest_int('estimator_min_samples_split', 2, 30),
                'min_samples_leaf': trial.suggest_int('estimator_min_samples_leaf', 1, 200),
                'max_features': trial.suggest_categorical('estimator_max_features', ['auto', 'sqrt', 'log2']),
                'subsample': trial.suggest_uniform('estimator_subsample', 0.1, 1),
            }
        }
        return trial, hyperparameters

    process_y = to_survival_y_records
    predict = predict_survival


class GBHist(GB):

    @staticmethod
    def get_estimator(X: DataFrame, verbose=0, advanced_impute=False, n_bins: int = 50):
        return DFPipeline(
            [
                *get_standard_steps(
                    TransformTarget(
                        GradientBoostingSurvivalAnalysis(random_state=RANDOM_STATE, verbose=verbose),
                        to_survival_y_records,
                    ),
                    X,
                ),
            ]
        )


class GBLimited(GB):

    @staticmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]:
        hyperparameters = {
            CROSS_VALIDATE_KEY: {
                'missing_fraction': trial.suggest_uniform(f'{CROSS_VALIDATE_KEY}_missing_fraction', 0.1, 1),
            },
            'estimator__inner': {
                'learning_rate': trial.suggest_uniform('estimator_learning_rate', 0, 1),
                'min_samples_leaf': trial.suggest_int('estimator_min_samples_leaf', 1, 200),
                'max_features': trial.suggest_categorical('estimator_max_features', ['auto', 'sqrt', 'log2']),
                'subsample': trial.suggest_uniform('estimator_subsample', 0.1, 1),
            }
        }
        return trial, hyperparameters


class RSF(Method):

    @staticmethod
    def get_estimator(X: DataFrame, verbose=0, advanced_impute=False) -> DFPipeline:
        return DFPipeline(
            get_standard_steps(
                TransformTarget(
                    RandomSurvivalForest(random_state=RANDOM_STATE, n_jobs=1),
                    to_survival_y_records,
                ),
                X,
            )
        )

    @staticmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]:
        hyperparameters = {
            'estimator__inner': {
                'n_estimators': trial.suggest_int('estimator_n_estimators', 5, 200),
                'max_depth': trial.suggest_int('estimator_max_depth', 1, 30),
                'min_samples_split': trial.suggest_int('estimator_min_samples_split', 2, 30),
                'min_samples_leaf': trial.suggest_int('estimator_min_samples_leaf', 1, 200),
                'max_features': trial.suggest_categorical('estimator_max_features', ['auto', 'sqrt', 'log2']),
                'oob_score': trial.suggest_categorical('estimator_oob_score', [True, False]),
            }
        }
        return trial, hyperparameters

    process_y = to_survival_y_records
    predict = predict_survival


def get_standard_steps(
    estimator: Estimator,
    X: DataFrame,
    advanced_impute: bool = False,
) -> List:
    return [
        *(get_mice_imputer(X) if advanced_impute else get_basic_imputer(X)),
        *get_encoder(X),
        ('estimator', estimator),
    ]


def get_preprocessing(X: DataFrame, standard_scaler: bool = False) -> Tuple:
    categorical_features, continuous_features = categorize_features(X)

    pipeline = [
        *([('scaler', DFStandardScaler())] if standard_scaler else []),
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


def transform_X(input):
    return numpy.array(input).astype('float32')


# class NNWrapper(CoxPH):
#     net: Optional[nn.Module]
#
#     def __init__(self, net=None, *args, **kwargs):
#         self.net = net
#         self.args = args
#         self.kwargs = kwargs
#
#     def fit(self, X, y, **kwargs):
#         if not self.net:
#             self.net = make_net(X.shape[1])
#
#         super().__init__(self.net, tt.optim.Adam, *self.args, **self.kwargs)
#         super().fit(X.astype('float32'), y)
#         return self
#
#
# def make_net(in_features: int) -> nn.Module:
#     num_nodes = [32, 32]
#     out_features = 1
#     batch_norm = True
#     dropout = 0.1
#     return tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
#
#
# def get_pycox_pipeline(X: DataFrame, verbose: int = 0):
#     preprocessing, categorical_features, continuous_features = get_preprocessing(X)
#
#     return DFPipeline(
#         [
#             *preprocessing,
#             ('standard', DFStandardScaler()),
#             ('to_numpy', FunctionTransformer(transform_X)),
#             ('estimator', NNWrapper()),
#         ],
#         # memory=CACHE_DIR
#     )
#


class DeepCoxMixtureMethod(Method):

    @staticmethod
    def get_estimator(X: DataFrame, verbose=True, advanced_impute=False, federated_groups: Dict[str, Index] = None):
        return DFPipeline(
            [
                ('scaler', DFStandardScaler()),
                *get_standard_steps(
                    DeepCoxMixtureAdapter(federated_groups=federated_groups), X, advanced_impute=advanced_impute
                ),
            ]
        )

    @staticmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]:
        hyperparameters = {
            CROSS_VALIDATE_KEY: {
                'missing_fraction': trial.suggest_uniform(f'{CROSS_VALIDATE_KEY}__missing_fraction', 0.1, 1),
            },
            'estimator': {
                'k': trial.suggest_int('estimator_k', 1, 5),
                'layers': [trial.suggest_int('layers_k', 3, 100)],
                'learning_rate': trial.suggest_loguniform('estimator_learning_rate', 0.0001, 1),
                'batch_size': trial.suggest_int('estimator_batch_size', 1, 100),
                'optimizer': trial.suggest_categorical('estimator_optimizer', ('Adam', 'RMSProp', 'SGD'))
            }
        }
        return trial, hyperparameters

    predict = predict_survival_dsm


class DeepCoxMixtureAdapter(DFWrapped, BaseEstimator, DeepCoxMixtures):
    model = None

    def __init__(
        self,
        k=3,
        layers=None,
        learning_rate=None,
        batch_size=None,
        optimizer=None,
        iters=100,
        federated_groups: Dict[str, Index] = None,
    ):
        super().__init__(k, layers)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.iters = iters
        self.federated_groups = federated_groups

    def fit(self, X, y, **kwargs):
        self.fitted_feature_names = self.get_fitted_feature_names(X)

        additional_kwargs = pipe(
            {
                **kwargs,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'optimizer': self.optimizer,
                'iters': self.iters,
            },
            reject_none_values,
        )

        super(DFWrapped, self).fit(
            X,
            y,
            federated_groups=self.federated_groups,
            **additional_kwargs,
        )

        return self

    def predict(self, X, t=365):
        return 1 - super().predict_survival(X.to_numpy(), t)

    def predict_survival(self, X, t):
        super().predict_survival(X.to_numpy(), t)

    def predict_survival_function(self, X):
        for i, (_, x) in enumerate(X.iterrows()):
            yield lambda t: self.predict_survival(X.iloc[i:i + 1], t)


class DeepCoxMixtureOriginalMethod(Method):

    @staticmethod
    def get_estimator(
        X: DataFrame,
        verbose=True,
        advanced_impute=False,
    ):
        return DFPipeline(
            [
                ('scaler', DFStandardScaler()),
                *get_standard_steps(DeepCoxMixturesOriginalAdapter(), X, advanced_impute=advanced_impute),
            ]
        )

    @staticmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]:
        hyperparameters = {
            CROSS_VALIDATE_KEY: {
                'missing_fraction': trial.suggest_uniform(f'{CROSS_VALIDATE_KEY}__missing_fraction', 0.1, 1),
            },
            'estimator': {
                'k': trial.suggest_int('estimator_k', 1, 5),
                'layers': [trial.suggest_int('layers_k', 3, 100)],
                'learning_rate': trial.suggest_loguniform('estimator_learning_rate', 0.0001, 1),
                'batch_size': trial.suggest_int('estimator_batch_size', 1, 100),
                'optimizer': trial.suggest_categorical('estimator_optimizer', ('Adam', 'RMSProp', 'SGD'))
            }
        }
        return trial, hyperparameters

    predict = predict_survival_dsm


class DeepCoxMixturesOriginalAdapter(DFWrapped, BaseEstimator, DeepCoxMixturesOriginal):
    model = None

    def __init__(self, k=3, layers=None, learning_rate=None, batch_size=None, optimizer=None, iters=100):
        super().__init__(k, layers)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.iters = iters

    def fit(self, X, y, **kwargs):
        self.fitted_feature_names = self.get_fitted_feature_names(X)
        t = y['data']['tte'].to_numpy().astype('float32')
        e = y['data']['label'].to_numpy()

        additional_kwargs = pipe(
            {
                **kwargs,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'optimizer': self.optimizer,
                'iters': self.iters,
            },
            reject_none_values,
        )
        super(DFWrapped, self).fit(X.to_numpy(), t, e, **additional_kwargs)
        return self

    def predict(self, X, t=365):
        return 1 - super().predict_survival(X.to_numpy(), t)

    def predict_survival(self, X, t):
        super().predict_survival(X.to_numpy(), t)

    def predict_survival_function(self, X):
        for i, (_, x) in enumerate(X.iterrows()):
            yield lambda t: self.predict_survival(X.iloc[i:i + 1], t)


class DeepSurvivalMachinesMethod(Method):

    @staticmethod
    def get_estimator(X: DataFrame, verbose=True, advanced_impute=False):
        return DFPipeline(
            [
                ('scaler', DFStandardScaler()),
                *get_standard_steps(DeepCoxMixtureAdapter(), X, advanced_impute=advanced_impute)
            ]
        )

    @staticmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]:
        hyperparameters = {
            CROSS_VALIDATE_KEY: {
                'missing_fraction': trial.suggest_uniform(f'{CROSS_VALIDATE_KEY}__missing_fraction', 0.1, 1),
            },
            'estimator': {
                'k': trial.suggest_int('estimator_k', 1, 5),
                'layers': [trial.suggest_int('layers_k', 3, 100)],
                'learning_rate': trial.suggest_loguniform('estimator_learning_rate', 0.0001, 1),
                'batch_size': trial.suggest_int('estimator_batch_size', 1, 100),
                'optimizer': trial.suggest_categorical('estimator_optimizer', ('Adam', 'RMSProp', 'SGD'))
            }
        }
        return trial, hyperparameters

    predict = predict_survival_dsm


class DeepSurvivalMachinesAdapter(DFWrapped, BaseEstimator, DeepSurvivalMachines):
    model = None

    def __init__(self, k=3, layers=None, learning_rate=None, batch_size=None, optimizer=None, iters=50):
        super().__init__(k, layers)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.iters = iters

    def fit(self, X, y, **kwargs):
        self.fitted_feature_names = self.get_fitted_feature_names(X)
        t = y['data']['tte'].to_numpy().astype('float32')
        e = y['data']['label'].to_numpy()

        additional_kwargs = pipe(
            {
                **kwargs,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'optimizer': self.optimizer,
                'iters': self.iters,
            },
            reject_none_values,
        )
        super(DFWrapped, self).fit(X.to_numpy(), t, e, **additional_kwargs)
        return self

    def predict(self, X, t=365):
        return 1 - super().predict_survival(X.to_numpy(), t)

    def predict_survival(self, X, t):
        super().predict_survival(X.to_numpy(), t)

    def predict_survival_function(self, X):
        for i, (_, x) in enumerate(X.iterrows()):
            yield lambda t: self.predict_survival(X.iloc[i:i + 1], t)


class DeepSurvMethod(Method):

    @staticmethod
    def get_estimator(X: DataFrame, verbose=True, advanced_impute=False):
        return DFPipeline(
            [
                ('scaler', DFStandardScaler()),
                *get_standard_steps(DeepSurvEstimator(), X, advanced_impute=advanced_impute)
            ]
        )

    @staticmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]:
        batch_size_limit = trial.suggest_categorical('estimator_batch_size_limit', [True, False])
        hyperparameters = {
            CROSS_VALIDATE_KEY: {
                'missing_fraction': trial.suggest_uniform(f'{CROSS_VALIDATE_KEY}__missing_fraction', 0.1, 1),
            },
            'estimator': {
                'layers': [trial.suggest_int('estimator_layers', 3, 100)],
                'learning_rate': trial.suggest_loguniform('estimator_learning_rate', 0.0001, 1),
                'lr_decay_rate': trial.suggest_loguniform('estimator_lr_decay_rate', 0.0001, 0.5),
                'drop': trial.suggest_uniform('estimator_drop', 0, 0.5),
                # TODO
                # 'batch_size': None
                # if not batch_size_limit else trial.suggest_int('estimator_batch_size', 1, 100),
                'optimizer': trial.suggest_categorical('estimator_optimizer', ('Adam', 'RMSprop', 'SGD'))
            }
        }
        return trial, hyperparameters

    predict = predict_predict


class DeepSurvEstimator(DFWrapped, BaseEstimator):
    model = None

    def __init__(
        self,
        layers=None,
        learning_rate=0.1,
        batch_size=None,
        optimizer='Adam',
        epochs=50,
        lr_decay_rate=3.579e-4,
        patience=30,
        drop=0.375,
        activation='SELU',
    ):

        if layers is None:
            self.layers = [100]
        else:
            self.layers = layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.epochs = epochs
        self.lr_decay_rate = lr_decay_rate
        self.patience = patience
        self.drop = drop
        self.activation = activation

    def fit(self, X, y, **kwargs):
        device = 'cuda'
        network = dict(
            drop=self.drop,
            norm=True,
            dims=[len(X.columns), *self.layers, 1],
            activation=self.activation,
            l2_reg=0,
        )
        # builds network|criterion|optimizer based on configuration
        self.model = DeepSurv(network).to(device)
        criterion = NegativeLogLikelihood(network).to(device)
        optimizer_class = eval('optim.{}'.format(self.optimizer))
        optimizer = optimizer_class(self.model.parameters(), lr=self.learning_rate)
        # constructs data loaders based on configuration
        train_loader = DataLoader(
            list(zip(
                X_to_pytorch(X),
                X_to_pytorch(y['data']['tte']),
                X_to_pytorch(y['data']['label']),
            )),
            batch_size=len(X) if self.batch_size is None else self.batch_size,
        )
        # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_dataset.__len__())
        # training
        best_c_index = 0
        flag = 0
        for epoch in range(1, self.epochs + 1):
            # adjusts learning rate
            lr = adjust_learning_rate(optimizer, epoch, self.learning_rate, self.lr_decay_rate)
            logger.info(f'Epoch {epoch}')
            # train step
            self.model.train()
            for X_batch, y_tte, y_label in train_loader:
                # makes predictions
                risk_pred = self.model(X_batch)
                train_loss = criterion(risk_pred, y_tte, y_label, self.model)
                # train_c = c_index(-risk_pred, y_tte, y_label)
                # updates parameters
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
            # valid step
            self.model.eval()

            for X, y, e in train_loader:
                # makes predictions
                with torch.no_grad():
                    risk_pred = self.model(X)
                    valid_loss = criterion(risk_pred, y, e, self.model)
                    valid_c = c_index(-risk_pred, y, e)
                    logger.info(f'c index:{valid_c}')
                    if best_c_index < valid_c:
                        best_c_index = valid_c
                        flag = 0
                        # saves the best model
                    else:
                        flag += 1
                        if flag >= self.patience:
                            logger.info('Early stopping')
                            return best_c_index
        return self

    def predict(self, X, t=365):
        y_pred = self.model(X_to_pytorch(X))
        return Series(y_pred.reshape(-1).tolist(), index=X.index)


class XGBClassifierMethod(Method, ABC):

    @staticmethod
    def get_estimator(X: DataFrame, verbose=0, advanced_impute=False):
        return DFPipeline(
            get_standard_steps(
                TransformerTarget(
                    XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='mlogloss'),
                    LabelEncoder(),
                ),
                X=X,
            )
        )

    predict = predict_proba
