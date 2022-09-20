from dataclasses import dataclass
from enum import auto
from itertools import starmap
from sklearn.mixture import GaussianMixture
from sksurv.tree import SurvivalTree

from deps.methods.pcp_hf import PooledCohort
from typing import Dict, Tuple, List, Type, Optional, Callable, Any

# noinspection PyUnresolvedReferences
import torch.optim as optim

from deps.methods.xgb import XGB
from hcve_lib.custom_types import Estimator, Method, StrEnum, Target, TrainTestIndex, Prediction
from hcve_lib.cv import CROSS_VALIDATE_KEY, configuration_to_params
from hcve_lib.data import to_survival_y_records, categorize_features
from hcve_lib.evaluation_functions import predict_survival, predict_survival_dsm, predict_proba
from hcve_lib.functional import pipe, reject_none
from hcve_lib.pipelines import TransformTarget
from hcve_lib.transformers import MiceForest
from hcve_lib.utils import remove_column_prefix, SurvivalResample, loc
from hcve_lib.wrapped_sklearn import DFCoxnetSurvivalAnalysis, DFPipeline, DFColumnTransformer, DFSimpleImputer, \
    DFOrdinalEncoder, DFStandardScaler, DFStacking, DFSurvivalTree
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from optuna import Trial
from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.preprocessing import FunctionTransformer

from deps.constants import RANDOM_STATE
from deps.data import auto_convert_category
from deps.methods.dcm import DeepCoxMixtures
from hcve_lib.pipelines import LifeTime
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.svm import FastKernelSurvivalSVM


def get_pipelines() -> Dict[str, Type[Method]]:
    return {
        'coxnet': CoxNetPipeline,
        'gb': GBPipeline,
        'dcm': DeepCoxMixturePipeline,
        'stacking': StackingPipeline,
        'stacking2': Stacking2Pipeline,
        'stacking_tree': StackingPipeline,
        'rsf': RSFPipeline,
        'xgb': XGBPipeline,
        'pcp_hf': PooledCohortPipeline,
        'svm': SVMPipeline,
        # TODO
        # 'stacking': Stacking,
        # 'gb_limited': GBLimited,
        # 'dcm_original': DeepCoxMixtureOriginalMethod,
        # 'dsm': DeepSurvivalMachinesMethod,
        # 'deep_surv': DeepSurvMethod,
        # 'dummy': Dummy,
    }


class ResamplingOption(StrEnum):
    NONE = auto()
    SUBSAMPLING = auto()
    OVERSAMPLING = auto()


class ImputeOption(StrEnum):
    SIMPLE = 'SIMPLE'
    MICE = 'MICE'


@dataclass
class PipelineConfiguration:
    impute: Optional[ImputeOption] = ImputeOption.SIMPLE
    resampling: Optional[ResamplingOption] = ResamplingOption.NONE
    standardize: bool = False
    lifetime: bool = False
    encode: bool = True
    age_range: Optional[Tuple] = (30, 80)

    def update(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)
        return self


def make_pipeline(
    estimators: List[Tuple[str, Estimator]],
    X: DataFrame,
    configuration: PipelineConfiguration = PipelineConfiguration(),
) -> DFPipeline:
    return DFPipeline(
        pipe(
            [
                *get_imputer(X, configuration.impute),
                get_resampling(configuration.resampling),
                *([('scaler', DFStandardScaler())] if configuration.standardize else ()),
                *(get_encoder(X) if configuration.encode else ()),
                *(
                    starmap(lambda name, estimator:
                            (name, LifeTime(estimator)), estimators) if configuration.lifetime else estimators
                ),
            ],
            reject_none,
            list,
        ),  # memory=memory,
    )


def get_imputer(
    X: DataFrame,
    impute: ImputeOption,
) -> List[Tuple[str, TransformerMixin]]:
    if impute == ImputeOption.MICE:
        return [('imputer', MiceForest(random_state=RANDOM_STATE, iterations=5))]
    elif impute == ImputeOption.SIMPLE:
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
                            DFPipeline([
                                ('imputer_mean', DFSimpleImputer(strategy='mean')),
                            ]),
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
    else:
        raise ValueError(f'Impute option \'{impute}\' is not valid')


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


def get_resampling(resampling: ResamplingOption) -> Tuple[str, ]:
    if resampling is ResamplingOption.OVERSAMPLING:
        return (
            'resample',
            SurvivalResample(RandomOverSampler()),
        )
    elif resampling is ResamplingOption.SUBSAMPLING:
        return (
            'resample',
            SurvivalResample(RandomUnderSampler()),
        )
    elif resampling is ResamplingOption.NONE:
        return None
    else:
        raise ValueError(f'Value \'{resampling}\' is unknown')


class TransformDTypesTransformDTypes(TransformerMixin):

    def __init__(
        self,
        categorical: List[str],
        continuous: List[str],
    ):
        self.categorical = categorical
        self.continuous = continuous

    def transform(self, X: DataFrame) -> DataFrame:
        for feature in self.categorical:
            if feature in X:
                X[feature] = X[feature].astype('object', copy=False)

        for feature in self.continuous:
            if feature in X:
                X[feature] = X[feature].astype('float', copy=False)


class AutoTransformDTypes(TransformerMixin):

    def fit(self, X, y):
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        return auto_convert_category(X)


class RandomSurvivalForestT(TransformTarget, RandomSurvivalForest):
    transform_callback: Callable[[Target], Any]

    def __init__(
        self,
        transform_callback: Callable[[Target], Any],
        n_estimators=100,
        max_depth=None,
        min_samples_split=6,
        min_samples_leaf=3,
        min_weight_fraction_leaf=0.,
        max_features="auto",
        max_leaf_nodes=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        max_samples=None,
    ):
        self.transform_callback = transform_callback

        RandomSurvivalForest.__init__(
            self,
            n_estimators,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            min_weight_fraction_leaf,
            max_features,
            max_leaf_nodes,
            bootstrap,
            oob_score,
            n_jobs,
            random_state,
            verbose,
            warm_start,
            max_samples,
        )

    @property
    def feature_importances_(self):
        raise Exception()


class RSFPipeline(Method):

    @staticmethod
    def get_estimator(
        X: DataFrame,
        random_state: int,
        configuration: PipelineConfiguration,
        verbose: bool = False,
    ) -> DFPipeline:
        return make_pipeline(
            [
                (
                    'estimator',
                    RandomSurvivalForestT(
                        transform_callback=to_survival_y_records, random_state=RANDOM_STATE, n_jobs=1
                    ),
                )
            ],
            X,
            configuration=configuration,
        )

    @staticmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]:
        hyperparameters = {
            'estimator': {
                'n_estimators': trial.suggest_int('estimator_n_estimators', 5, 200),
                'max_depth': trial.suggest_int('estimator_max_depth', 1, 4),
                'min_samples_split': trial.suggest_int('estimator_min_samples_split', 2, 30),
                'min_samples_leaf': trial.suggest_int('estimator_min_samples_leaf', 1, 200),
                'max_features': trial.suggest_categorical('estimator_max_features', ['auto', 'sqrt', 'log2']),
                'oob_score': trial.suggest_categorical('estimator_oob_score', [True, False]),
            }
        }
        return trial, hyperparameters

    process_y = to_survival_y_records
    predict = predict_survival


class SurvivalTreeT(SurvivalTree):
    transform_callback: Callable[[Target], Any]

    def __init__(
        self,
        transform_callback: Callable[[Target], Any],
        splitter="best",
        max_depth=None,
        min_samples_split=6,
        min_samples_leaf=3,
        min_weight_fraction_leaf=0.,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None
    ):
        self.transform_callback = transform_callback
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes

    def fit(self, X, y):
        return super().fit(X, self.transform_callback(y))

    def score(self, X, y):
        return super().score(X, self.transform_callback(y))

    def predict_proba(self, X, *args, **kwargs):
        return self.predict(X)


class CoxnetSurvivalAnalysisT(CoxnetSurvivalAnalysis):
    transform_callback: Callable[[Target], Any]

    def __init__(
        self,
        transform_callback: Callable[[Target], Any],
        n_alphas=100,
        alphas=None,
        alpha_min_ratio="auto",
        l1_ratio=0.5,
        penalty_factor=None,
        normalize=False,
        copy_X=True,
        tol=1e-7,
        max_iter=100000,
        verbose=False,
        fit_baseline_model=False
    ):
        self.transform_callback = transform_callback
        super().__init__(
            n_alphas,
            alphas,
            alpha_min_ratio,
            l1_ratio,
            penalty_factor,
            normalize,
            copy_X,
            tol,
            max_iter,
            verbose,
            fit_baseline_model,
        )

    def fit(self, X, y):
        return super().fit(X, self.transform_callback(y))

    def score(self, X, y):
        return super().score(X, self.transform_callback(y))

    def predict_proba(self, X, *args, **kwargs):
        return self.predict(X)


class GradientBoostingSurvivalAnalysisT(GradientBoostingSurvivalAnalysis):
    transform_callback: Callable[[Target], Any]

    def __init__(
        self,
        transform_callback: Callable[[Target], Any],
        loss="coxph",
        learning_rate=0.1,
        n_estimators=100,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.,
        max_depth=3,
        min_impurity_decrease=0.,
        random_state=None,
        max_features=None,
        max_leaf_nodes=None,
        subsample=1.0,
        dropout_rate=0.0,
        verbose=0,
        ccp_alpha=0.0
    ):
        self.transform_callback = transform_callback
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            verbose=verbose,
            ccp_alpha=ccp_alpha,
            dropout_rate=dropout_rate
        )

    def fit(self, X, y):
        return super().fit(X, self.transform_callback(y))

    def score(self, X, y):
        return super().score(X, self.transform_callback(y))

    def predict_proba(self, X, *args, **kwargs):
        return self.predict(X)


class DFCoxnetSurvivalAnalysisT(DFCoxnetSurvivalAnalysis):
    transform_callback: Callable[[Target], Any]

    def __init__(
        self,
        transform_callback: Callable[[Target], Any],
        n_alphas=100,
        alphas=None,
        alpha_min_ratio="auto",
        l1_ratio=0.5,
        penalty_factor=None,
        normalize=False,
        copy_X=True,
        tol=1e-7,
        max_iter=100000,
        verbose=False,
        fit_baseline_model=False
    ):
        self.transform_callback = transform_callback
        super().__init__(
            n_alphas,
            alphas,
            alpha_min_ratio,
            l1_ratio,
            penalty_factor,
            normalize,
            copy_X,
            tol,
            max_iter,
            verbose,
            fit_baseline_model,
        )

    def fit(self, X, y):
        return super().fit(X, self.transform_callback(y))

    def score(self, X, y):
        return super().score(X, self.transform_callback(y))

    def predict_proba(self, X, *args, **kwargs):
        return self.predict(X)


class CoxNetPipeline(Method):

    @classmethod
    def get_estimator(
        cls,
        X: DataFrame,
        random_state: int,
        configuration: PipelineConfiguration,
        verbose: bool = False,
    ):
        return make_pipeline(
            [cls.get_final_estimator(verbose=verbose)], X, configuration=configuration.update(standardize=True)
        )

    @classmethod
    def get_final_estimator(cls, verbose: bool = False):
        return (
            'estimator',
            DFCoxnetSurvivalAnalysisT(
                to_survival_y_records,
                fit_baseline_model=True,
                verbose=verbose,
                n_alphas=1,
            ),
        )

    @staticmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]:
        hyperparameters = {
            CROSS_VALIDATE_KEY: {
                'missing_fraction': trial.suggest_uniform(f'{CROSS_VALIDATE_KEY}__missing_fraction', 0.1, 1),
            },
            'estimator': {
                'l1_ratio': 1 - trial.suggest_loguniform('estimator_l1_ratio', 0.1, 1),
                'alphas': [trial.suggest_loguniform('estimator_alphas', 10**-2, 1)],
                'n_alphas': 1,
            }
        }
        return trial, hyperparameters

    process_y = to_survival_y_records
    predict = predict_survival


class GBPipeline(Method):

    @staticmethod
    def get_estimator(
        X: DataFrame,
        random_state: int,
        configuration: PipelineConfiguration,
        verbose=0,
    ) -> DFPipeline:
        return make_pipeline(
            [
                (
                    'estimator',
                    GradientBoostingSurvivalAnalysisT(
                        transform_callback=to_survival_y_records,
                        random_state=RANDOM_STATE,
                        verbose=verbose,
                    ),
                )
            ],
            X,
            configuration=configuration,
        )

    @staticmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]:
        hyperparameters = {
            CROSS_VALIDATE_KEY: {
                'missing_fraction': trial.suggest_uniform(f'{CROSS_VALIDATE_KEY}_missing_fraction', 0.1, 1),
            },
            'estimator': {
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


class FastKernelSurvivalSVMT(FastKernelSurvivalSVM):

    def __init__(
        self,
        transform_callback: Callable[[Target], Any],
        alpha=1,
        rank_ratio=1.0,
        fit_intercept=False,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        max_iter=20,
        verbose=False,
        tol=None,
        optimizer=None,
        random_state=None,
        timeit=False
    ):
        self.transform_callback = transform_callback
        super().__init__(
            alpha=alpha,
            rank_ratio=rank_ratio,
            fit_intercept=fit_intercept,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            max_iter=max_iter,
            verbose=verbose,
            tol=tol,
            optimizer=optimizer,
            random_state=random_state,
            timeit=timeit,
        )

    def fit(self, X, y):
        return super().fit(X, self.transform_callback(y))

    def set_params(self, **kwargs):
        print(kwargs)
        return super().set_params(**kwargs)

    def score(self, X, y):
        return super().score(X, self.transform_callback(y))

    def predict_proba(self, X, *args, **kwargs):
        return self.predict(X)


class SVMPipeline(Method):

    @staticmethod
    def get_estimator(
        X: DataFrame,
        random_state: int,
        configuration: PipelineConfiguration,
        verbose=0,
    ) -> DFPipeline:
        return make_pipeline(
            [
                (
                    'estimator',
                    FastKernelSurvivalSVMT(
                        transform_callback=to_survival_y_records,
                        random_state=RANDOM_STATE,
                        verbose=verbose > 0,
                    ),
                )
            ],
            X,
            configuration=configuration.update(standardize=True),
        )

    @staticmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]:
        hyperparameters = {
            CROSS_VALIDATE_KEY: {
                'missing_fraction': trial.suggest_uniform(f'{CROSS_VALIDATE_KEY}_missing_fraction', 0.1, 1),
            },
            'estimator': {
                'alpha': trial.suggest_uniform('estimator_alpha', 0, 1000),
            }
        }
        hyperparameters['estimator']['kernel'] = trial.suggest_categorical(
            'estimator_kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
        )

        if hyperparameters['estimator']['kernel'] == 'rbf':
            hyperparameters['estimator']['degree'] = trial.suggest_int('estimator_degree', 1, 3)

        return trial, hyperparameters

    process_y = to_survival_y_records
    predict = predict_survival


class DFSurvivalTreeT(DFSurvivalTree):
    transform_callback: Callable[[Target], Any]

    def __init__(
        self,
        transform_callback: Callable[[Target], Any],
        splitter="best",
        max_depth=None,
        min_samples_split=6,
        min_samples_leaf=3,
        min_weight_fraction_leaf=0.,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
    ):
        self.transform_callback = transform_callback
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes

    def fit(self, X, y):
        return super().fit(X, self.transform_callback(y))

    def score(self, X, y):
        return super().score(X, self.transform_callback(y))

    def predict_proba(self, X, *args, **kwargs):
        return self.predict(X)


class GaussianMixtureT(GaussianMixture):

    def predict_proba(self, X):
        return self.predict(X)


class TreePipeline(Method):

    @staticmethod
    def get_estimator(
        X: DataFrame,
        random_state: int,
        configuration: PipelineConfiguration,
        verbose=0,
    ) -> DFPipeline:
        return make_pipeline(
            [('estimator', DFSurvivalTreeT(
                transform_callback=to_survival_y_records,
                random_state=RANDOM_STATE,
            ))],
            X,
            configuration=configuration,
        )

    @staticmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]:
        hyperparameters = {
            CROSS_VALIDATE_KEY: {
                'missing_fraction': trial.suggest_uniform(f'{CROSS_VALIDATE_KEY}_missing_fraction', 0.1, 1),
            },
            'estimator': {
                'splitter': trial.suggest_categorical('estimator_splitter', ['best', 'random']),
                'max_features': trial.suggest_categorical('estimator_max_features', ['auto', 'sqrt', 'log2']),
                'max_depth': trial.suggest_int('estimator_max_depth', 1, 10),
                'min_samples_leaf': trial.suggest_int('estimator_min_samples_leaf', 1, 200),
            }
        }
        return trial, hyperparameters

    process_y = to_survival_y_records
    predict = predict_survival


def drop_pcp(X):
    print(f'{X=}')
    return X


class StackingPipeline(Method):

    @classmethod
    def get_estimator(
        cls,
        X: DataFrame,
        random_state: int,
        configuration: PipelineConfiguration,
        verbose: bool = False,
    ):
        stack = make_pipeline(
            [
                (
                    'estimator',
                    DFStacking(
                        CoxnetSurvivalAnalysisT(transform_callback=to_survival_y_records),
                        [(name, estimator) for name, estimator in cls.get_base_learners()],
                    ),
                )
            ],
            X,
            configuration.update(encode=False, standardize=False)
        )

        return stack

    @staticmethod
    def get_base_learners() -> List[Tuple[str, Callable]]:
        return [
            (
                'coxnet',
                DFPipeline(
                    [
                        ('preprocessor', DFStandardScaler()),
                        (
                            'internal',
                            CoxnetSurvivalAnalysisT(transform_callback=to_survival_y_records, fit_baseline_model=True)
                        )
                    ]
                ),
            ),
            ('gb', GradientBoostingSurvivalAnalysisT(transform_callback=to_survival_y_records)),
            ('pcp_hf', PooledCohort()),
            ('gm', GaussianMixtureT(n_components=5)),
        ]

    @staticmethod
    def get_base_learners_optuna() -> List[Tuple[str, Callable]]:
        return [('estimator__coxnet__internal', CoxNetPipeline.optuna), ('estimator__gb', GBPipeline.optuna)]

    @classmethod
    def get_meta_optuna(cls, *args, **kwargs):
        return CoxNetPipeline.optuna(*args, **kwargs)

    @classmethod
    def optuna(cls, trial: Trial) -> Tuple[Trial, Dict]:
        _, hyperparameters_meta = cls.get_meta_optuna(trial)

        hyperparameters = {
            'estimator__meta_estimator': hyperparameters_meta['estimator'],
        }

        for name, optuna in cls.get_base_learners_optuna():
            hyperparameters[name] = optuna(trial)[1]['estimator']

        params = configuration_to_params(hyperparameters)

        params[CROSS_VALIDATE_KEY] = {
            'missing_fraction': trial.suggest_uniform(f'{CROSS_VALIDATE_KEY}__missing_fraction', 0.1, 1),
        }

        return trial, params

    process_y = to_survival_y_records
    predict = predict_proba


class Stacking2Pipeline(Method):

    @classmethod
    def get_estimator(
        cls,
        X: DataFrame,
        random_state: int,
        configuration: PipelineConfiguration,
        verbose: bool = False,
    ):
        stack = make_pipeline(
            [
                (
                    'estimator',
                    DFStacking(
                        CoxnetSurvivalAnalysisT(transform_callback=to_survival_y_records),
                        [(name, estimator) for name, estimator in cls.get_base_learners()],
                    ),
                )
            ],
            X,
            configuration.update(encode=False, standardize=False)
        )

        return stack

    @staticmethod
    def get_base_learners() -> List[Tuple[str, Callable]]:
        return [
            (
                'coxnet',
                DFPipeline(
                    [
                        ('preprocessor', DFStandardScaler()),
                        (
                            'internal',
                            CoxnetSurvivalAnalysisT(transform_callback=to_survival_y_records, fit_baseline_model=True)
                        )
                    ]
                ),
            ),
            ('gb', GradientBoostingSurvivalAnalysisT(transform_callback=to_survival_y_records)),
        ]

    @staticmethod
    def get_base_learners_optuna() -> List[Tuple[str, Callable]]:
        return [('estimator__coxnet__internal', CoxNetPipeline.optuna), ('estimator__gb', GBPipeline.optuna)]

    @classmethod
    def get_meta_optuna(cls, *args, **kwargs):
        return CoxNetPipeline.optuna(*args, **kwargs)

    @classmethod
    def optuna(cls, trial: Trial) -> Tuple[Trial, Dict]:
        _, hyperparameters_meta = cls.get_meta_optuna(trial)

        hyperparameters = {
            'estimator__meta_estimator': hyperparameters_meta['estimator'],
        }

        for name, optuna in cls.get_base_learners_optuna():
            hyperparameters[name] = optuna(trial)[1]['estimator']

        params = configuration_to_params(hyperparameters)

        params[CROSS_VALIDATE_KEY] = {
            'missing_fraction': trial.suggest_uniform(f'{CROSS_VALIDATE_KEY}__missing_fraction', 0.1, 1),
        }

        return trial, params

    process_y = to_survival_y_records
    predict = predict_proba


class DeepCoxMixturePipeline(Method):

    @staticmethod
    def get_estimator(
        X: DataFrame,
        random_state: int,
        configuration: PipelineConfiguration,
        verbose=0,
    ) -> DFPipeline:
        return make_pipeline(
            [
                ('scaler', DFStandardScaler()),
                ('estimator', DeepCoxMixtures()),
            ],
            X,
            configuration,
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


class XGBPipeline(Method):

    @staticmethod
    def get_estimator(
        X: DataFrame,
        random_state: int,
        configuration: PipelineConfiguration,
        verbose=0,
    ) -> DFPipeline:
        return make_pipeline(
            [
                ('scaler', DFStandardScaler()),
                ('estimator', XGB()),
            ],
            X,
        )

    @staticmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]:
        hp = {
            CROSS_VALIDATE_KEY: {
                'missing_fraction': trial.suggest_uniform(f'{CROSS_VALIDATE_KEY}__missing_fraction', 0.1, 1),
            },
            'estimator': {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
                "lambda_": trial.suggest_loguniform("lambda", 1e-8, 1.0),
                "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
            }
        }

        if hp['estimator']["booster"] == "gbtree" or hp['estimator']["booster"] == "dart":
            hp['estimator']["max_depth"] = trial.suggest_int("max_depth", 1, 9)
            hp['estimator']["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
            hp['estimator']["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
            hp['estimator']["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

        if hp['estimator']["booster"] == "dart":
            hp['estimator']["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            hp['estimator']["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            hp['estimator']["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
            hp['estimator']["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

        return trial, hp

    predict = predict_survival_dsm


#
#
# TODO
# ====
# class GBHist(GB):
#
#     @staticmethod
#     def get_estimator(X: DataFrame, verbose=0, advanced_impute=False, n_bins: int = 50):
#         return DFPipeline(
#             [
#                 *get_standard_steps(
#                     TransformTarget(
#                         GradientBoostingSurvivalAnalysis(random_state=RANDOM_STATE, verbose=verbose),
#                         to_survival_y_records,
#                     ),
#                     X,
#                 ),
#             ]
#         )
#
#
# class GBLimited(GB):
#
#     @staticmethod
#     def optuna(trial: Trial) -> Tuple[Trial, Dict]:
#         hyperparameters = {
#             CROSS_VALIDATE_KEY: {
#                 'missing_fraction': trial.suggest_uniform(f'{CROSS_VALIDATE_KEY}_missing_fraction', 0.1, 1),
#             },
#             'estimator__inner': {
#                 'learning_rate': trial.suggest_uniform('estimator_learning_rate', 0, 1),
#                 'min_samples_leaf': trial.suggest_int('estimator_min_samples_leaf', 1, 200),
#                 'max_features': trial.suggest_categorical('estimator_max_features', ['auto', 'sqrt', 'log2']),
#                 'subsample': trial.suggest_uniform('estimator_subsample', 0.1, 1),
#             }
#         }
#         return trial, hyperparameters
#
#

#
#
# class Dummy(Method):
#
#     @staticmethod
#     def get_estimator(
#         X: DataFrame,
#         random_state: int,
#         verbose=0,
#         advanced_impute=False,
#     ):
#         return DFPipeline([
#             *get_basic_imputer(X),
#             ('estimator', DummyEstimator()),
#         ])
#
#     @staticmethod
#     def optuna(trial: Trial) -> Tuple[Trial, Dict]:
#         raise Exception('Not possible to optimize')
#
#     @staticmethod
#     def predict(
#         X: DataFrame,
#         y: Target,
#         split: TrainTestIndex,
#         model: 'DummyEstimator',
#         random_state: int,
#         method: Type[Method],
#         time: int = 3 * 365,
#     ) -> Prediction:
#         X_test = loc(
#             split[1],
#             X,
#             ignore_not_present=True,
#         )
#
#         return Prediction(
#             split=split,
#             X_columns=X.columns.tolist(),
#             y_column=y['name'],
#             y_score=Series(
#                 model.predict(X_test),
#                 index=X_test.index,
#             ),
#             y_proba={time: model.predict_survival_function(X)(time)},
#             model=model,
#             random_state=random_state,
#             method=method,
#         )
#
#
# class DummyEstimator(BaseEstimator):
#     y: Target
#
#     def fit(self, X: DataFrame, y: Target):
#         self.y = y
#
#     def predict(self, X: DataFrame) -> Series:
#         return 1 - LabelEncoder().fit_transform(self.y['data']['label']).mean()
#
#     def predict_survival_function(
#         self,
#         X: DataFrame,
#     ) -> Callable[[int], Series]:
#         return lambda time: Series(
#             [1 - binarize_event(time, self.y['data']).mean()] * len(X),
#             index=X.index,
#         )
#
#


def pl(d, *args):
    print(len(d.columns))
    print(len(d))


class PooledCohortPipeline(Method):

    @staticmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]:
        raise Exception('Not possible to optimize')

    @staticmethod
    def get_estimator(
        X: DataFrame,
        random_state: int,
        configuration: PipelineConfiguration,
        verbose=0,
    ) -> DFPipeline:
        return make_pipeline(
            [
                ('estimator', PooledCohort()),
            ],
            X,
            configuration.update(encode=False),
        )

    @staticmethod
    def predict(
        X: DataFrame,
        y: Target,
        split: TrainTestIndex,
        model: Estimator,
        method: Type['Method'],
        random_state: int,
    ) -> Prediction:
        return Prediction(
            split=split,
            X_columns=X.columns.tolist(),
            y_column=y['name'],
            y_score=model.predict(loc(split[1], X)),
            model=model,
            random_state=random_state,
            method=method,
        )


# class PooledCohort(BaseEstimator):
#
#     def fit(self, *args, **kwargs):
#         pass
#
#     def predict(self, X):
#         X_ = X.copy()
#
#         def predict_(row: Series) -> float:
#             if row['AGE'] < 30:
#                 return np.nan
#
#             bsug_adj = row['GLU'] * 18.018
#             row['CHOL'] = row['CHOL'] * 38.67
#             row['HDL'] = row['HDL'] * 38.67
#             bmi = row['BMI']
#             sex = row['SEX']
#             qrs = row['QRS']
#             trt_ht = row['TRT_AH']
#             if row['SMK'] == 0 or row['SMK'] == 2:
#                 csmk = 0
#             elif row['SMK'] == 1:
#                 csmk = 1
#             ln_age = log(row['AGE'])
#             ln_age_sq = ln_age**2
#             ln_tchol = log(row['CHOL'])
#             ln_hchol = log(row['HDL'])
#             ln_sbp = log(row['SBP'])
#             hdm = row['DIABETES']
#             ln_agesbp = ln_age * ln_sbp
#             ln_agecsmk = ln_age * csmk
#             ln_bsug = log(bsug_adj)
#             ln_bmi = log(bmi)
#             ln_agebmi = ln_age * ln_bmi
#             ln_qrs = log(qrs)
#
#             if (sex == 1) and (trt_ht == 0): coeff_sbp = 0.91
#             if (sex == 1) and (trt_ht == 1): coeff_sbp = 1.03
#             if (sex == 2) and (trt_ht == 0): coeff_sbp = 11.86
#             if (sex == 2) and (trt_ht == 1): coeff_sbp = 12.95
#             if (sex == 2) and (trt_ht == 0): coeff_agesbp = -2.73
#             if (sex == 2) and (trt_ht == 1): coeff_agesbp = -2.96
#
#             if (sex == 1) and (hdm == 0): coeff_bsug = 0.78
#             if (sex == 1) and (hdm == 1): coeff_bsug = 0.90
#             if (sex == 2) and (hdm == 0): coeff_bsug = 0.91
#             if (sex == 2) and (hdm == 1): coeff_bsug = 1.04
#
#             if sex == 1:
#                 IndSum = ln_age * 41.94 + ln_age_sq * (
#                     -0.88
#                 ) + ln_sbp * coeff_sbp + csmk * 0.74 + ln_bsug * coeff_bsug + ln_tchol * 0.49 + ln_hchol * (
#                     -0.44
#                 ) + ln_bmi * 37.2 + ln_agebmi * (-8.83) + ln_qrs * 0.63
#
#                 HF_risk = 100 * (1 - (0.98752**exp(IndSum - 171.5)))
#             elif sex == 2:
#                 IndSum = ln_age * 20.55 + ln_sbp * coeff_sbp + ln_agesbp * coeff_agesbp + csmk * 11.02 + ln_agecsmk * (
#                     -2.50
#                 ) + ln_bsug * coeff_bsug + ln_hchol * (-0.07) + ln_bmi * 1.33 + ln_qrs * 1.06
#                 HF_risk = 100 * (1 - (0.99348**exp(IndSum - 99.73)))
#             return HF_risk
#
#         return X.apply(predict_, axis=1).dropna()
#
#
# class Stacking(Method):
#
#     @staticmethod
#     def get_estimator(X: DataFrame, verbose=0, advanced_impute=False):
#         return get_standard_steps(
#             TransformTarget(
#                 DFStacking(
#                     DecisionTreeRegressor(),
#                     [('coxnet', DFCoxnetSurvivalAnalysis), ('gb', GradientBoostingSurvivalAnalysis)]
#                 ),
#                 to_survival_y_records,
#             ),
#             X,
#             advanced_impute=advanced_impute,
#         )
#
#     @staticmethod
#     def optuna(trial: Trial) -> Tuple[Trial, Dict]:
#         hyperparameters = {
#             CROSS_VALIDATE_KEY: {
#                 'missing_fraction': trial.suggest_uniform(f'{CROSS_VALIDATE_KEY}__missing_fraction', 0.1, 1),
#             },
#             'estimator__inner': {
#                 'l1_ratio': 1 - trial.suggest_loguniform('estimator_n_alphas', 0.1, 1),
#             }
#         }
#         return trial, hyperparameters
#
#     process_y = to_survival_y_records
#     predict = predict_survival
#
#
# def transform_X(input):
#     return numpy.array(input).astype('float32')
#
#
# # class NNWrapper(CoxPH):
# #     net: Optional[nn.Module]
# #
# #     def __init__(self, net=None, *args, **kwargs):
# #         self.net = net
# #         self.args = args
# #         self.kwargs = kwargs
# #
# #     def fit(self, X, y, **kwargs):
# #         if not self.net:
# #             self.net = make_net(X.shape[1])
# #
# #         super().__init__(self.net, tt.optim.Adam, *self.args, **self.kwargs)
# #         super().fit(X.astype('float32'), y)
# #         return self
# #
# #
# # def make_net(in_features: int) -> nn.Module:
# #     num_nodes = [32, 32]
# #     out_features = 1
# #     batch_norm = True
# #     dropout = 0.1
# #     return tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
# #
# #
# # def get_pycox_pipeline(X: DataFrame, verbose: int = 0):
# #     preprocessing, categorical_features, continuous_features = get_preprocessing(X)
# #
# #     return DFPipeline(
# #         [
# #             *preprocessing,
# #             ('standard', DFStandardScaler()),
# #             ('to_numpy', FunctionTransformer(transform_X)),
# #             ('estimator', NNWrapper()),
# #         ],
# #         # memory=CACHE_DIR
# #     )
# #
#
#

#
# class DeepCoxMixtureOriginalMethod(Method):
#
#     @staticmethod
#     def get_estimator(
#         X: DataFrame,
#         verbose=True,
#         advanced_impute=False,
#     ):
#         return DFPipeline(
#             [
#                 ('scaler', DFStandardScaler()),
#                 *get_standard_steps(DeepCoxMixturesOriginalAdapter(), X, advanced_impute=advanced_impute),
#             ]
#         )
#
#     @staticmethod
#     def optuna(trial: Trial) -> Tuple[Trial, Dict]:
#         hyperparameters = {
#             CROSS_VALIDATE_KEY: {
#                 'missing_fraction': trial.suggest_uniform(f'{CROSS_VALIDATE_KEY}__missing_fraction', 0.1, 1),
#             },
#             'estimator': {
#                 'k': trial.suggest_int('estimator_k', 1, 5),
#                 'layers': [trial.suggest_int('layers_k', 3, 100)],
#                 'learning_rate': trial.suggest_loguniform('estimator_learning_rate', 0.0001, 1),
#                 'batch_size': trial.suggest_int('estimator_batch_size', 1, 100),
#                 'optimizer': trial.suggest_categorical('estimator_optimizer', ('Adam', 'RMSProp', 'SGD'))
#             }
#         }
#         return trial, hyperparameters
#
#     predict = predict_survival_dsm
#
#
# class DeepCoxMixturesOriginalAdapter(DFWrapped, BaseEstimator, DeepCoxMixturesOriginal):
#     model = None
#
#     def __init__(self, k=3, layers=None, learning_rate=None, batch_size=None, optimizer=None, iters=100):
#         super().__init__(k, layers)
#         self.learning_rate = learning_rate
#         self.batch_size = batch_size
#         self.optimizer = optimizer
#         self.iters = iters
#
#     def fit(self, X, y, **kwargs):
#         self.fit_feature_names = self.get_fitted_feature_names(X)
#         t = y['data']['tte'].to_numpy().astype('float32')
#         e = y['data']['label'].to_numpy()
#
#         additional_kwargs = pipe(
#             {
#                 **kwargs,
#                 'learning_rate': self.learning_rate,
#                 'batch_size': self.batch_size,
#                 'optimizer': self.optimizer,
#                 'iters': self.iters,
#             },
#             reject_none_values,
#         )
#         super(DFWrapped, self).fit(X.to_numpy(), t, e, **additional_kwargs)
#         return self
#
#     def predict(self, X, t=365):
#         return 1 - super().predict_survival(X.to_numpy(), t)
#
#     def predict_survival(self, X, t):
#         super().predict_survival(X.to_numpy(), t)
#
#     def predict_survival_function(self, X):
#         for i, (_, x) in enumerate(X.iterrows()):
#             yield lambda t: self.predict_survival(X.iloc[i:i + 1], t)

#
# class DeepSurvivalMachinesMethod(Method):
#
#     @staticmethod
#     def get_estimator(X: DataFrame, verbose=True, advanced_impute=False):
#         return DFPipeline(
#             [
#                 ('scaler', DFStandardScaler()),
#                 *get_standard_steps(DeepCoxMixtureAdapter(), X, advanced_impute=advanced_impute)
#             ]
#         )
#
#     @staticmethod
#     def optuna(trial: Trial) -> Tuple[Trial, Dict]:
#         hyperparameters = {
#             CROSS_VALIDATE_KEY: {
#                 'missing_fraction': trial.suggest_uniform(f'{CROSS_VALIDATE_KEY}__missing_fraction', 0.1, 1),
#             },
#             'estimator': {
#                 'k': trial.suggest_int('estimator_k', 1, 5),
#                 'layers': [trial.suggest_int('layers_k', 3, 100)],
#                 'learning_rate': trial.suggest_loguniform('estimator_learning_rate', 0.0001, 1),
#                 'batch_size': trial.suggest_int('estimator_batch_size', 1, 100),
#                 'optimizer': trial.suggest_categorical('estimator_optimizer', ('Adam', 'RMSProp', 'SGD'))
#             }
#         }
#         return trial, hyperparameters
#
#     predict = predict_survival_dsm
#
#
# class DeepSurvivalMachinesAdapter(DFWrapped, BaseEstimator, DeepSurvivalMachines):
#     model = None
#
#     def __init__(self, k=3, layers=None, learning_rate=None, batch_size=None, optimizer=None, iters=50):
#         super().__init__(k, layers)
#         self.learning_rate = learning_rate
#         self.batch_size = batch_size
#         self.optimizer = optimizer
#         self.iters = iters
#
#     def fit(self, X, y, **kwargs):
#         self.fit_feature_names = self.get_fitted_feature_names(X)
#         t = y['data']['tte'].to_numpy().astype('float32')
#         e = y['data']['label'].to_numpy()
#
#         additional_kwargs = pipe(
#             {
#                 **kwargs,
#                 'learning_rate': self.learning_rate,
#                 'batch_size': self.batch_size,
#                 'optimizer': self.optimizer,
#                 'iters': self.iters,
#             },
#             reject_none_values,
#         )
#         super(DFWrapped, self).fit(X.to_numpy(), t, e, **additional_kwargs)
#         return self
#
#     def predict(self, X, t=365):
#         return 1 - super().predict_survival(X.to_numpy(), t)
#
#     def predict_survival(self, X, t):
#         super().predict_survival(X.to_numpy(), t)
#
#     def predict_survival_function(self, X):
#         for i, (_, x) in enumerate(X.iterrows()):
#             yield lambda t: self.predict_survival(X.iloc[i:i + 1], t)
#
#
# class DeepSurvMethod(Method):
#
#     @staticmethod
#     def get_estimator(X: DataFrame, verbose=True, advanced_impute=False):
#         return DFPipeline(
#             [
#                 ('scaler', DFStandardScaler()),
#                 *get_standard_steps(DeepSurvEstimator(), X, advanced_impute=advanced_impute)
#             ]
#         )
#
#     @staticmethod
#     def optuna(trial: Trial) -> Tuple[Trial, Dict]:
#         batch_size_limit = trial.suggest_categorical('estimator_batch_size_limit', [True, False])
#         hyperparameters = {
#             CROSS_VALIDATE_KEY: {
#                 'missing_fraction': trial.suggest_uniform(f'{CROSS_VALIDATE_KEY}__missing_fraction', 0.1, 1),
#             },
#             'estimator': {
#                 'layers': [trial.suggest_int('estimator_layers', 3, 100)],
#                 'learning_rate': trial.suggest_loguniform('estimator_learning_rate', 0.0001, 1),
#                 'lr_decay_rate': trial.suggest_loguniform('estimator_lr_decay_rate', 0.0001, 0.5),
#                 'drop': trial.suggest_uniform('estimator_drop', 0, 0.5),
#                 # TODO
#                 # 'batch_size': None
#                 # if not batch_size_limit else trial.suggest_int('estimator_batch_size', 1, 100),
#                 'optimizer': trial.suggest_categorical('estimator_optimizer', ('Adam', 'RMSprop', 'SGD'))
#             }
#         }
#         return trial, hyperparameters
#
#     predict = predict_predict
#
#
# class DeepSurvEstimator(DFWrapped, BaseEstimator):
#     model = None
#
#     def __init__(
#         self,
#         layers=None,
#         learning_rate=0.1,
#         batch_size=None,
#         optimizer='Adam',
#         epochs=50,
#         lr_decay_rate=3.579e-4,
#         patience=30,
#         drop=0.375,
#         activation='SELU',
#     ):
#
#         if layers is None:
#             self.layers = [100]
#         else:
#             self.layers = layers
#         self.learning_rate = learning_rate
#         self.batch_size = batch_size
#         self.optimizer = optimizer
#         self.epochs = epochs
#         self.lr_decay_rate = lr_decay_rate
#         self.patience = patience
#         self.drop = drop
#         self.activation = activation
#
#     def fit(self, X, y, **kwargs):
#         device = 'cuda'
#         network = dict(
#             drop=self.drop,
#             norm=True,
#             dims=[len(X.columns), *self.layers, 1],
#             activation=self.activation,
#             l2_reg=0,
#         )
#         # builds network|criterion|optimizer based on configuration
#         self.model = DeepSurv(network).to(device)
#         criterion = NegativeLogLikelihood(network).to(device)
#         optimizer_class = eval('optim.{}'.format(self.optimizer))
#         optimizer = optimizer_class(self.model.parameters(), lr=self.learning_rate)
#         # constructs data loaders based on configuration
#         train_loader = DataLoader(
#             list(zip(
#                 X_to_pytorch(X),
#                 X_to_pytorch(y['data']['tte']),
#                 X_to_pytorch(y['data']['label']),
#             )),
#             batch_size=len(X) if self.batch_size is None else self.batch_size,
#         )
#         # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_dataset.__len__())
#         # training
#         best_c_index = 0
#         flag = 0
#         for epoch in range(1, self.epochs + 1):
#             # adjusts learning rate
#             lr = adjust_learning_rate(optimizer, epoch, self.learning_rate, self.lr_decay_rate)
#             logger.info(f'Epoch {epoch}')
#             # train step
#             self.model.train()
#             for X_batch, y_tte, y_label in train_loader:
#                 # makes predictions
#                 risk_pred = self.model(X_batch)
#                 train_loss = criterion(risk_pred, y_tte, y_label, self.model)
#                 # train_c = c_index(-risk_pred, y_tte, y_label)
#                 # updates parameters
#                 optimizer.zero_grad()
#                 train_loss.backward()
#                 optimizer.step()
#             # valid step
#             self.model.eval()
#
#             for X, y, e in train_loader:
#                 # makes predictions
#                 with torch.no_grad():
#                     risk_pred = self.model(X)
#                     valid_loss = criterion(risk_pred, y, e, self.model)
#                     valid_c = c_index(-risk_pred, y, e)
#                     logger.info(f'c index:{valid_c}')
#                     if best_c_index < valid_c:
#                         best_c_index = valid_c
#                         flag = 0
#                         # saves the best model
#                     else:
#                         flag += 1
#                         if flag >= self.patience:
#                             logger.info('Early stopping')
#                             return best_c_index
#         return self
#
#     def predict(self, X, t=365):
#         y_pred = self.model(X_to_pytorch(X))
#         return Series(y_pred.reshape(-1).tolist(), index=X.index)
#
#
# class XGBClassifierMethod(Method, ABC):
#
#     @staticmethod
#     def get_estimator(X: DataFrame, verbose=0, advanced_impute=False):
#         return DFPipeline(
#             get_standard_steps(
#                 TransformerTarget(
#                     XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='mlogloss'),
#                     LabelEncoder(),
#                 ),
#                 X=X,
#             )
#         )
#
#     predict = predict_proba
