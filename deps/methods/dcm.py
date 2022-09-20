# noinspection PyUnresolvedReferences
import torch.optim as optim
from hcve_lib.functional import pipe, reject_none_values
from hcve_lib.wrapped_sklearn import DFWrapped
from sklearn.base import BaseEstimator

from dsm_original import DeepCoxMixtures as DeepCoxMixtures_


class DeepCoxMixtures(DFWrapped, BaseEstimator, DeepCoxMixtures_):
    model = None

    def __init__(self, k=3, layers=None, learning_rate=None, batch_size=None, optimizer=None, iters=100):
        super().__init__(k, layers)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.iters = iters

    def fit(self, X, y, **kwargs):
        self.fitted_feature_names = self.get_fit_features(X)
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
        return super().predict_survival(X.to_numpy(), t)

    def predict_survival_function(self, X):
        for i, (_, x) in enumerate(X.iterrows()):
            yield lambda t: self.predict_survival(X.iloc[i:i + 1], t)
