from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

from optuna import Trial
from pandas import DataFrame

from hcve_lib.custom_types import Target, Estimator


class MethodDefinition(ABC):

    @staticmethod
    @abstractmethod
    def process_y(y: DataFrame) -> Any:
        ...

    @staticmethod
    @abstractmethod
    def get_estimator(X: DataFrame, verbose=0, advanced_impute=False):
        ...

    @staticmethod
    @abstractmethod
    def optuna(trial: Trial) -> Tuple[Trial, Dict]:
        ...

    @staticmethod
    @abstractmethod
    def predict(
        X_train: DataFrame,
        y_train: DataFrame,
        X_test: DataFrame,
        y_true: Target,
        model: Estimator,
    ):
        ...
